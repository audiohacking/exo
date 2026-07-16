"""Tests for StreamingPrefillLayers — the per-layer KV-ready hook that
disaggregated prefill streaming is built on."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.disaggregated.streaming_prefill import (
    StreamingPrefillLayers,
)


class _MockAttnLayer(nn.Module):
    """Mimics a real transformer decoder layer: on each call, appends seq_len
    tokens' worth of KV into `cache`, the way a real attention layer does."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self._seq_len = seq_len

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        if cache is not None:
            with mx.stream(mx.Device(mx.cpu)):
                new_k = mx.zeros((1, 1, self._seq_len, 4))
                new_v = mx.zeros((1, 1, self._seq_len, 4))
            existing_k, existing_v = cache.keys, cache.values
            if existing_k is None or existing_v is None:
                cache.keys, cache.values = new_k, new_v
                cache.offset = self._seq_len
            else:
                cache.keys = mx.concatenate([existing_k, new_k], axis=2)
                cache.values = mx.concatenate([existing_v, new_v], axis=2)
                cache.offset += self._seq_len
        return x


class _NoCacheLayer(nn.Module):
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x


class _MockInnerModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = layers


class _MockModel(nn.Module):
    """Matches get_inner_model()'s expectation of a `.model` attribute."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.model = _MockInnerModel(layers)


def test_fires_callback_per_layer() -> None:
    layers: list[nn.Module] = [_MockAttnLayer(seq_len=3) for _ in range(4)]
    model = _MockModel(layers)
    caches = [KVCache() for _ in layers]

    fired: list[int] = []
    with StreamingPrefillLayers(model, fired.append):
        x = mx.zeros((1, 4))
        for i, layer in enumerate(model.model.layers):
            x = layer(x, cache=caches[i])
            mx.eval(x)

    assert fired == [0, 1, 2, 3]
    for c in caches:
        assert c.offset == 3


def test_replaces_layers_while_active_and_restores_on_exit() -> None:
    originals: list[nn.Module] = [_MockAttnLayer(seq_len=1) for _ in range(2)]
    model = _MockModel(originals)

    with StreamingPrefillLayers(model, lambda _i: None):
        for orig, wrapped in zip(originals, model.model.layers, strict=True):
            assert wrapped is not orig

    for orig, restored in zip(originals, model.model.layers, strict=True):
        assert orig is restored


def test_restores_original_layers_on_exception() -> None:
    originals: list[nn.Module] = [_MockAttnLayer(seq_len=1)]
    model = _MockModel(originals)

    with pytest.raises(RuntimeError), StreamingPrefillLayers(model, lambda _i: None):
        raise RuntimeError("boom mid-prefill")

    assert model.model.layers[0] is originals[0]


def test_skips_layer_without_cache_kwarg() -> None:
    """A layer whose signature has no `cache` param should just not fire the
    callback — never crash the forward pass."""
    model = _MockModel([_NoCacheLayer()])
    fired: list[int] = []
    with StreamingPrefillLayers(model, fired.append):
        x = mx.zeros((1, 4))
        _ = model.model.layers[0](x)

    assert fired == []
