"""Tests for the streaming disaggregated-prefill orchestration in
batch_generator.py — _serve_prefill_streaming and _streaming_prefill_enabled.

run_prefill_for_request is monkeypatched (mirrors test_serve_prefill.py's approach
of patching mlx_prefill) so these exercise the wire-protocol/threading orchestration
without needing a real model forward pass.
"""

import io
from collections.abc import Callable
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

import exo.worker.runner.llm_inference.batch_generator as batch_generator_mod
from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.types import KVCacheType, Model

_FAKE_MODEL = cast(Model, cast(object, None))
_FAKE_TOKENIZER = cast(TokenizerWrapper, cast(object, None))


def _make_kv_layer_cache(seq_len: int, n_heads: int = 2, head_dim: int = 4) -> KVCache:
    mx.random.seed(0)
    c = KVCache()
    with mx.stream(mx.Device(mx.cpu)):
        c.keys = (mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10).astype(
            mx.bfloat16
        )
        c.values = (
            mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
        ).astype(mx.bfloat16)
        mx.eval(c.keys, c.values)
    c.offset = seq_len
    return c


def _decode_all(payload: bytes) -> tuple[Header, list[KVChunk | ArraysState], int]:
    buf = io.BytesIO(payload)
    hdr = read_header(buf)
    msgs: list[KVChunk | ArraysState] = []
    total = 0
    while True:
        msg = read_message(buf)
        if msg is None:
            break
        if isinstance(msg, Done):
            total = msg.total_tokens
            break
        assert isinstance(msg, (KVChunk, ArraysState))
        msgs.append(msg)
    return hdr, msgs, total


def _request(n_tokens: int) -> PrefillRequest:
    return PrefillRequest(
        request_id="r1", model_id="m", token_ids=list(range(n_tokens)), start_pos=0
    )


def test_streaming_prefill_enabled_parses_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("1", "true", "True", "yes", "YES"):
        monkeypatch.setenv("EXO_STREAMING_PREFILL", value)
        assert batch_generator_mod._streaming_prefill_enabled() is True  # pyright: ignore[reportPrivateUsage]
    for value in ("0", "false", "", "no"):
        monkeypatch.setenv("EXO_STREAMING_PREFILL", value)
        assert batch_generator_mod._streaming_prefill_enabled() is False  # pyright: ignore[reportPrivateUsage]
    monkeypatch.delenv("EXO_STREAMING_PREFILL", raising=False)
    assert batch_generator_mod._streaming_prefill_enabled() is False  # pyright: ignore[reportPrivateUsage]


def test_serve_prefill_streaming_sends_chunk_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_layers = 3
    seq_len = 5

    def fake_run_prefill_for_request(
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        group: mx.distributed.Group | None,
        kv_prefix_cache: object,
        request: PrefillRequest,
        on_layer_ready: Callable[[int, KVCacheType], None] | None = None,
    ) -> KVCacheType:
        cache: list[KVCache] = []
        for i in range(n_layers):
            cache.append(_make_kv_layer_cache(seq_len))
            if on_layer_ready is not None:
                on_layer_ready(i, cache)
        return cache

    monkeypatch.setattr(
        batch_generator_mod, "run_prefill_for_request", fake_run_prefill_for_request
    )

    buf = io.BytesIO()
    batch_generator_mod._serve_prefill_streaming(  # pyright: ignore[reportPrivateUsage]
        model=_FAKE_MODEL,
        tokenizer=_FAKE_TOKENIZER,
        group=None,
        kv_prefix_cache=None,
        request=_request(seq_len + 2),
        wfile=buf,
    )

    hdr, msgs, total = _decode_all(buf.getvalue())
    assert hdr.num_layers == n_layers
    assert total == seq_len
    kv_msgs = [m for m in msgs if isinstance(m, KVChunk)]
    assert sorted(m.layer_idx for m in kv_msgs) == [0, 1, 2]
    assert all(m.num_tokens == seq_len for m in kv_msgs)


def test_serve_prefill_streaming_falls_back_to_final_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If on_layer_ready never fires (e.g. an unsupported layer/cache shape, or a
    full prefix-cache hit with nothing to prefill), correctness must not depend on
    it — the final flush pass over the returned cache must still send everything."""
    n_layers = 2
    seq_len = 4

    def fake_run_prefill_for_request(
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        group: mx.distributed.Group | None,
        kv_prefix_cache: object,
        request: PrefillRequest,
        on_layer_ready: Callable[[int, KVCacheType], None] | None = None,
    ) -> KVCacheType:
        return [_make_kv_layer_cache(seq_len) for _ in range(n_layers)]

    monkeypatch.setattr(
        batch_generator_mod, "run_prefill_for_request", fake_run_prefill_for_request
    )

    buf = io.BytesIO()
    batch_generator_mod._serve_prefill_streaming(  # pyright: ignore[reportPrivateUsage]
        model=_FAKE_MODEL,
        tokenizer=_FAKE_TOKENIZER,
        group=None,
        kv_prefix_cache=None,
        request=_request(seq_len + 2),
        wfile=buf,
    )

    hdr, msgs, total = _decode_all(buf.getvalue())
    assert hdr.num_layers == n_layers
    assert total == seq_len
    kv_msgs = [m for m in msgs if isinstance(m, KVChunk)]
    assert sorted(m.layer_idx for m in kv_msgs) == [0, 1]


def test_serve_prefill_streaming_sends_arrays_cache_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq_len = 3

    def fake_run_prefill_for_request(
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        group: mx.distributed.Group | None,
        kv_prefix_cache: object,
        request: PrefillRequest,
        on_layer_ready: Callable[[int, KVCacheType], None] | None = None,
    ) -> KVCacheType:
        kv = _make_kv_layer_cache(seq_len)
        arr = ArraysCache(size=1)
        with mx.stream(mx.Device(mx.cpu)):
            state = mx.ones((2,))
            mx.eval(state)
        arr.state = [state]
        cache: list[KVCache | ArraysCache] = [kv, arr]
        # Real _StreamingKVLayer only fires on_layer_ready for KVCache/RotatingKVCache
        # entries — ArraysCache (SSM/hybrid state) never gets an early callback, so
        # it must be picked up by the final flush pass, exactly once.
        if on_layer_ready is not None:
            on_layer_ready(0, cache)
        return cache

    monkeypatch.setattr(
        batch_generator_mod, "run_prefill_for_request", fake_run_prefill_for_request
    )

    buf = io.BytesIO()
    batch_generator_mod._serve_prefill_streaming(  # pyright: ignore[reportPrivateUsage]
        model=_FAKE_MODEL,
        tokenizer=_FAKE_TOKENIZER,
        group=None,
        kv_prefix_cache=None,
        request=_request(seq_len + 2),
        wfile=buf,
    )

    _, msgs, _ = _decode_all(buf.getvalue())
    arrays_msgs = [m for m in msgs if isinstance(m, ArraysState)]
    assert len(arrays_msgs) == 1
    assert arrays_msgs[0].layer_idx == 1


def test_serve_prefill_delegates_to_streaming_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_STREAMING_PREFILL", "1")
    called: list[bool] = []

    def fake_streaming(**_kwargs: object) -> None:
        called.append(True)

    monkeypatch.setattr(
        batch_generator_mod,
        "_serve_prefill_streaming",
        fake_streaming,
    )

    batch_generator_mod._serve_prefill(  # pyright: ignore[reportPrivateUsage]
        model=_FAKE_MODEL,
        tokenizer=_FAKE_TOKENIZER,
        group=None,
        kv_prefix_cache=None,
        request=_request(5),
        wfile=io.BytesIO(),
    )
    assert called == [True]


def test_serve_prefill_uses_bulk_path_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_STREAMING_PREFILL", raising=False)
    streaming_called: list[bool] = []

    def fake_streaming(**_kwargs: object) -> None:
        streaming_called.append(True)

    monkeypatch.setattr(
        batch_generator_mod,
        "_serve_prefill_streaming",
        fake_streaming,
    )

    def fake_run_prefill_for_request(**_kwargs: object) -> KVCacheType:
        return [_make_kv_layer_cache(3)]

    monkeypatch.setattr(
        batch_generator_mod, "run_prefill_for_request", fake_run_prefill_for_request
    )

    buf = io.BytesIO()
    batch_generator_mod._serve_prefill(  # pyright: ignore[reportPrivateUsage]
        model=_FAKE_MODEL,
        tokenizer=_FAKE_TOKENIZER,
        group=None,
        kv_prefix_cache=None,
        request=_request(5),
        wfile=buf,
    )
    assert streaming_called == []
    hdr, _, total = _decode_all(buf.getvalue())
    assert hdr.num_layers == 1
    assert total == 3
