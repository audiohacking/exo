import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
    DEFAULT_NUM_DRAFT_TOKENS,
    EXO_ADAPTIVE_DRAFT_TOKENS,
    EXO_DRAFTER_MIN_OUTPUT_TOKENS,
    EXO_NUM_DRAFT_TOKENS,
    BatchGenerator,
    SequentialGenerator,
    parse_env_int,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .generator.coupled_drafter import is_coupled_drafter_dispatchable
from .generator.drafter import EXO_DRAFT_MODE_ENV, parse_draft_mode
from .types import Model
from .utils_mlx import (
    CoupledDrafter,
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    # Standard external drafter (independent sibling LM sharing the target's
    # tokenizer), loaded by ``load_mlx_items`` when the model card declares
    # ``drafter_model_ids`` and the placement is single-device.
    draft_model: Model | None = None
    draft_model_id: ModelId | None = None
    # Coupled (mtp/dflash) drafter loaded via mlx-vlm. Mutually exclusive with
    # ``draft_model`` at the loader level: ``load_mlx_items`` tries the coupled
    # path first when the card declares ``coupled_drafter`` and falls back to
    # the standard external drafter only on coupled-load failure.
    coupled_drafter: CoupledDrafter | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
            self.draft_model,
            self.draft_model_id,
            self.coupled_drafter,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        # Drop drafters BEFORE the target / tokenizer / group: coupled
        # drafters bind to the target's input embeddings, and standard
        # drafters can hold a weak reference into the target's tokenizer.
        with contextlib.suppress(NameError, AttributeError):
            del self.draft_model
        with contextlib.suppress(NameError, AttributeError):
            del self.coupled_drafter
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)
        drafter_kv_prefix_cache: KVPrefixCache | None = (
            KVPrefixCache(self.group) if self.draft_model is not None else None
        )

        device_rank = 0 if self.group is None else self.group.rank()

        coupled_drafter_dispatchable = (
            self.coupled_drafter is not None
            and is_coupled_drafter_dispatchable(self.coupled_drafter.kind)
        )
        any_drafter_loaded = (
            self.draft_model is not None or coupled_drafter_dispatchable
        )
        configured_draft_mode = parse_draft_mode(
            os.environ.get(EXO_DRAFT_MODE_ENV),
            default="model" if any_drafter_loaded else "none",
        )
        allow_request_drafting = os.environ.get(
            "EXO_ALLOW_REQUEST_DRAFTING", ""
        ).lower() in {"1", "true", "yes"}
        is_single_device = self.group is None or self.group.size() == 1

        # Asymmetric drafter placement (drafter on a separate MLX rank) is not
        # ported to this fork — only collocated (single-device or symmetric
        # tensor-parallel) drafting is supported.
        drafting_can_run_here = is_single_device or coupled_drafter_dispatchable
        drafter_loaded_will_run = any_drafter_loaded and configured_draft_mode != "none"
        force_sequential_for_drafter = drafting_can_run_here and (
            drafter_loaded_will_run
            or allow_request_drafting
            or configured_draft_mode == "pipelined"
        )
        drafter_loaded_but_explicitly_disabled = (
            drafting_can_run_here
            and any_drafter_loaded
            and configured_draft_mode == "none"
            and not allow_request_drafting
        )

        if os.environ.get("EXO_NO_BATCH") or force_sequential_for_drafter:
            if force_sequential_for_drafter:
                if allow_request_drafting and not any_drafter_loaded:
                    logger.info(
                        "using SequentialGenerator (EXO_ALLOW_REQUEST_DRAFTING set; "
                        "BatchGenerator has no spec-decoding hook for request "
                        "overrides)"
                    )
                elif coupled_drafter_dispatchable:
                    assert self.coupled_drafter is not None  # narrowed by gate
                    logger.info(
                        f"using SequentialGenerator (coupled drafter loaded: "
                        f"{self.coupled_drafter.model_id} kind={self.coupled_drafter.kind!r}; "
                        f"draft_mode={configured_draft_mode!r}; BatchGenerator "
                        f"has no spec-decoding hook for coupled MTP/DFlash)"
                    )
                else:
                    logger.info(
                        f"using SequentialGenerator (draft_mode={configured_draft_mode!r}; "
                        f"BatchGenerator has no spec-decoding hook)"
                    )
            else:
                logger.info("using SequentialGenerator (batching disabled)")

            num_draft_tokens = parse_env_int(
                EXO_NUM_DRAFT_TOKENS, DEFAULT_NUM_DRAFT_TOKENS
            )
            drafter_min_output_tokens = parse_env_int(
                EXO_DRAFTER_MIN_OUTPUT_TOKENS,
                DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
                minimum=0,
            )
            adaptive_draft_tokens = os.environ.get(
                EXO_ADAPTIVE_DRAFT_TOKENS, ""
            ).lower() in {"1", "true", "yes"}
            if force_sequential_for_drafter:
                logger.info(
                    f"speculative decoding: mode={configured_draft_mode}, "
                    f"K={num_draft_tokens} (adaptive={adaptive_draft_tokens}), "
                    f"skip_drafter_when_max_tokens<={drafter_min_output_tokens}"
                )

            max_concurrent_tasks = EXO_MAX_CONCURRENT_REQUESTS
            if max_concurrent_tasks > 1:
                logger.info(
                    f"SequentialGenerator round-robin concurrency: "
                    f"max_concurrent_tasks={max_concurrent_tasks} "
                    f"(EXO_MAX_CONCURRENT_REQUESTS)"
                )

            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                draft_model=self.draft_model,
                draft_model_id=self.draft_model_id,
                coupled_drafter=self.coupled_drafter,
                drafter_kv_prefix_cache=drafter_kv_prefix_cache,
                num_draft_tokens=num_draft_tokens,
                drafter_min_output_tokens=drafter_min_output_tokens,
                adaptive_draft_tokens=adaptive_draft_tokens,
            )
        else:
            if not drafting_can_run_here and (
                any_drafter_loaded or allow_request_drafting
            ):
                logger.info(
                    f"using BatchGenerator (drafting unavailable on multi-device "
                    f"runner: group.size={self.group.size() if self.group is not None else 1}; "
                    f"mlx_generate would demote draft_mode='none' anyway, keeping "
                    f"batching for throughput)"
                )
            elif drafter_loaded_but_explicitly_disabled:
                logger.info(
                    "using BatchGenerator (drafter loaded but EXO_DRAFT_MODE=none; "
                    "keeping batching for throughput)"
                )
            else:
                logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
