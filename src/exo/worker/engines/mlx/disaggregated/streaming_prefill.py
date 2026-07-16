from collections.abc import Callable
from inspect import signature
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    get_inner_model,
    get_layers,
)
from exo.worker.runner.bootstrap import logger


class _StreamingKVLayer(CustomMlxLayer):
    """Wraps one transformer decoder layer so that, immediately after it computes its
    KV slice for this forward call, `on_layer_ready(layer_idx)` fires — letting the
    caller start streaming that layer's new KV bytes out while later layers are still
    computing (mirrors the PipelineLastLayer send-as-you-go pattern in auto_parallel.py,
    applied to the disaggregated-prefill KV path instead of the ring transport).

    Only ever installed for the duration of a single disaggregated-prefill request; the
    original layer is always restored afterwards via StreamingPrefillLayers.
    """

    def __init__(
        self,
        original_layer: object,
        layer_idx: int,
        on_layer_ready: Callable[[int], None],
    ):
        super().__init__(cast(nn.Module, original_layer))
        self._layer_idx = layer_idx
        self._on_layer_ready = on_layer_ready
        try:
            self._sig = signature(cast(Callable[..., object], original_layer).__call__)
        except (AttributeError, ValueError, TypeError):
            self._sig = None

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        output = self.original_layer(x, *args, **kwargs)
        if self._sig is not None:
            try:
                cache = self._sig.bind_partial(x, *args, **kwargs).arguments.get(
                    "cache", None
                )
                if cache is not None:
                    entry = cache[0] if hasattr(cache, "caches") else cache  # type: ignore
                    if isinstance(entry, (KVCache, RotatingKVCache)):
                        mx.eval(entry.keys, entry.values)
                        self._on_layer_ready(self._layer_idx)
            except Exception as e:
                # Best-effort: an unsupported layer/cache shape just doesn't stream
                # early — its KV still goes out in the caller's final flush pass.
                logger.opt(exception=e).debug(
                    f"Streaming prefill: layer {self._layer_idx} hook failed, "
                    "will fall back to non-streamed send"
                )
        return output


class StreamingPrefillLayers:
    """Context manager that temporarily wraps every transformer layer of `model` so
    on_layer_ready(layer_idx) fires as soon as that layer's KV cache is extended,
    instead of waiting for the whole forward pass (all layers, all chunks) to finish.
    Original layers are always restored on exit, even on error.
    """

    def __init__(self, model: nn.Module, on_layer_ready: Callable[[int], None]):
        self._model = model
        self._on_layer_ready = on_layer_ready
        self._originals: list[object] = []
        self._inner: nn.Module | None = None

    def __enter__(self) -> "StreamingPrefillLayers":
        inner = get_inner_model(self._model)
        layers = get_layers(inner)
        self._inner = inner
        self._originals = list(layers)
        for i in range(len(layers)):
            layers[i] = _StreamingKVLayer(layers[i], i, self._on_layer_ready)
        if hasattr(inner, "layers"):
            inner.layers = layers
        else:
            inner.h = layers
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._inner is None:
            return
        if hasattr(self._inner, "layers"):
            self._inner.layers = self._originals
        else:
            self._inner.h = self._originals
