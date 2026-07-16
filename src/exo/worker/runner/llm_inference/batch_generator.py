import itertools
import os
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import BinaryIO

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.protocol import (
    DType,
    Header,
    write_done,
    write_header,
    write_message,
)
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import KVPrefixCache, cache_length
from exo.worker.engines.mlx.disaggregated.adapter import (
    build_kv_chunk_for_entry,
    send_arrays_cache_entry,
    wire_dtype_from_cache,
    write_cache_to_wire,
)
from exo.worker.engines.mlx.disaggregated.serve import (
    compute_target_offset,
    run_prefill_for_request,
)
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_all_gather_tasks,
    mx_any,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers, map_responses_to_chunks
from .tool_parsers import ToolParser

# Bounds how many chunk sends can be queued to the background writer thread before the
# prefill forward pass blocks waiting for one to finish. Keeps memory bounded for very
# long prompts and gives natural backpressure if the network is the bottleneck.
_STREAMING_PREFILL_MAX_INFLIGHT = 2


def _streaming_prefill_enabled() -> bool:
    return os.getenv("EXO_STREAMING_PREFILL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _serve_prefill_streaming(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
    wfile: BinaryIO,
) -> None:
    """Stream each layer's KV cache slice out as soon as it's computed, instead of
    waiting for the whole prefill to finish (the default, EXO_STREAMING_PREFILL=1
    behavior). Overlaps DGX-side compute with the network transfer to the decode node.

    Falls back to a final flush pass over every layer after prefill completes, so
    correctness never depends on the per-layer hook firing for every layer (e.g. an
    unsupported layer/cache shape, or a full prefix-cache hit with nothing to prefill
    at all, degrade gracefully to a single bulk-style send at the end).
    """
    target_offset = compute_target_offset(len(request.token_ids))
    last_sent: dict[int, int] = {}
    sent_arrays: set[int] = set()
    header_written = False
    pending: deque[Future[None]] = deque()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefill-send")

    def _drain_one() -> None:
        pending.popleft().result()

    def _submit(fn: Callable[[], None]) -> None:
        if len(pending) >= _STREAMING_PREFILL_MAX_INFLIGHT:
            _drain_one()
        pending.append(executor.submit(fn))

    def _ensure_header(cache: KVCacheType, dtype: DType) -> None:
        nonlocal header_written
        if header_written:
            return
        write_header(
            wfile,
            Header(
                request_id=request.request_id,
                model_id=request.model_id,
                num_layers=len(cache),
                dtype=dtype,
                start_pos=request.start_pos,
            ),
        )
        header_written = True

    def _send_layer(layer_idx: int, cache: KVCacheType) -> None:
        entry = cache[layer_idx]
        match entry:
            case QuantizedKVCache() | CacheList() | DeepseekV4Cache():
                raise NotImplementedError
            case ArraysCache():
                if layer_idx in sent_arrays:
                    return
                sent_arrays.add(layer_idx)
                _ensure_header(cache, wire_dtype_from_cache(cache))

                def _write_arrays() -> None:
                    send_arrays_cache_entry(wfile, entry, layer_idx)
                    wfile.flush()

                _submit(_write_arrays)
            case KVCache() | RotatingKVCache():
                start = last_sent.get(layer_idx, request.start_pos)
                chunk = build_kv_chunk_for_entry(
                    entry,
                    layer_idx,
                    dtype=wire_dtype_from_cache(cache),
                    start_pos=start,
                    max_tokens=target_offset,
                )
                if chunk is None:
                    return
                _ensure_header(cache, chunk.dtype)
                last_sent[layer_idx] = start + chunk.num_tokens

                def _write_chunk() -> None:
                    write_message(wfile, chunk)
                    wfile.flush()

                _submit(_write_chunk)

    def _on_layer_ready(layer_idx: int, cache: KVCacheType) -> None:
        try:
            _send_layer(layer_idx, cache)
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Streaming prefill: failed to send layer {layer_idx} incrementally "
                f"for request_id={request.request_id}, will retry in final flush"
            )

    try:
        cache = run_prefill_for_request(
            model=model,
            tokenizer=tokenizer,
            group=group,
            kv_prefix_cache=kv_prefix_cache,
            request=request,
            on_layer_ready=_on_layer_ready,
        )

        for layer_idx in range(len(cache)):
            _send_layer(layer_idx, cache)

        while pending:
            _drain_one()

        # Only reached if nothing streamed and the final flush had nothing to send
        # either (e.g. a full prefix-cache hit — client already has everything).
        _ensure_header(cache, wire_dtype_from_cache(cache))
        total_tokens = max(
            0, min(cache_length(cache), target_offset) - request.start_pos
        )
        write_done(wfile, total_tokens)
        wfile.flush()
    finally:
        executor.shutdown(wait=True)


def _serve_prefill(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
    wfile: BinaryIO,
) -> None:
    if _streaming_prefill_enabled():
        _serve_prefill_streaming(
            model=model,
            tokenizer=tokenizer,
            group=group,
            kv_prefix_cache=kv_prefix_cache,
            request=request,
            wfile=wfile,
        )
        return
    cache = run_prefill_for_request(
        model=model,
        tokenizer=tokenizer,
        group=group,
        kv_prefix_cache=kv_prefix_cache,
        request=request,
    )
    write_cache_to_wire(
        wfile,
        cache,
        request_id=request.request_id,
        model_id=request.model_id,
        start_pos=request.start_pos,
    )


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: (
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Iterator[GenerationChunk | None],
        ]
        | None
    ) = field(default=None, init=False)

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return map(
                    lambda task: (task, CancelledResponse()), self._cancelled_tasks
                )

        assert self._active is not None

        task, gen, queue, output_generator = self._active
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        try:
            response = next(gen)
            queue.push(response)
            # drain potentially many responses every time
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

        except (StopIteration, PrefillCancelled):
            output.append((task.task_id, FinishedResponse()))
            self._active = None
            if self._queue:
                self._start_next()

        except Exception as e:
            self._send_error(task, e)
            self._active = None
            raise

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(
                output,
                map(lambda task: (task, CancelledResponse()), self._cancelled_tasks),
            ),
        )

    def _start_next(self) -> None:
        task = self._queue.popleft()
        try:
            gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        self._active = (task, gen, queue, output_generator)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        _serve_prefill(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
            wfile=wfile,
        )


@dataclass(eq=False)
class BatchGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _gen: ExoBatchGenerator = field(init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        if not self._queue:
            self.agree_on_tasks()

        # Submit any queued tasks to the engine
        while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            try:
                uid = self._start_task(task)
            except PrefillCancelled:
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator: Iterator[GenerationChunk | None] = map(
                    lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
                )
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(self.tokenizer, task.task_params),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                )
            self._active_tasks[uid] = (task, queue, output_generator)

        if not self._gen.has_work:
            return self._apply_cancellations()

        results = self._gen.step()

        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            task, queue, output_generator = self._active_tasks[uid]
            queue.push(response)
            # If a generator fails to parse for some reason and returns early, we should not crash
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            # check if original response was terminal and append a Finished()
            if response.finish_reason is not None:
                output.append((task.task_id, FinishedResponse()))
                del self._active_tasks[uid]

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(output, self._apply_cancellations()),
        )

    def _apply_cancellations(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, CancelledResponse]] = []

        for uid, (task, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, CancelledResponse()))
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, CancelledResponse()))

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _start_task(self, task: TextGeneration) -> int:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

                self.agree_on_tasks()

        return self._gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        _serve_prefill(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
            wfile=wfile,
        )
