import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast, final

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import VisionProcessor

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path, resolve_existing_model
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.auto_parallel import (
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    DFlashHooksNotImplementedError as _DFlashHooksNotImplementedError,
)
from exo.worker.runner.bootstrap import logger


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_HOSTS_JSON"] = hosts_json
                os.environ["MLX_RANK"] = str(rank)
                # os.environ["MLX_RING_VERBOSE"] = "1"  # NOTE: we don't use it enough to care (turn on again if need to)

                group = mx.distributed.init(backend="ring", strict=True)

                # Eagerly start TcpRelay server on CUDA nodes
                import platform

                is_linux_gpu = (
                    platform.system() == "Linux"
                    and mx.default_device().type == mx.DeviceType.gpu
                )
                logger.info(
                    f"CUDA check: os={platform.system()}, device={mx.default_device()}, is_linux_gpu={is_linux_gpu}"
                )
                if is_linux_gpu:
                    try:
                        from exo.worker.engines.mlx.auto_parallel import _get_tcp_relay

                        relay = _get_tcp_relay()
                        relay._ensure_server()
                        logger.info(
                            f"CUDA TcpRelay server started on port {relay._tcp_port}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to start TcpRelay: {e}")

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                # Use RDMA connectivity matrix
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: mx.distributed.Group | None,
) -> Generator[
    ModelLoadingResponse,
    None,
    tuple[
        Model,
        TokenizerWrapper,
        "VisionProcessor | None",
        Model | None,
        ModelId | None,
        "CoupledDrafter | None",
    ],
]:
    target_card = bound_instance.bound_shard.model_card
    target_size = get_weights_size(bound_instance.bound_shard)

    # Pre-include drafter size in the wired-memory limit so the OS doesn't
    # page out drafter weights between requests. The limit is configured once,
    # before loading the target, so the decision has to happen here.
    drafter_bytes = 0
    if not _drafter_disabled_by_env():
        if target_card.coupled_drafter is not None:
            drafter_bytes = _coupled_drafter_weight_size_bytes(
                target_card.coupled_drafter
            )
        elif group is None and target_card.drafter_model_ids:
            chosen = _select_drafter_id(
                list(target_card.drafter_model_ids), _drafter_preference()
            )
            if chosen is not None:
                drafter_bytes = _drafter_weight_size_bytes(chosen)
    set_wired_limit_for_model(target_size + Memory.from_bytes(drafter_bytes))

    drafter_model: Model | None = None
    drafter_id: ModelId | None = None
    coupled_drafter: CoupledDrafter | None = None

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Eval layers one by one for progress reporting
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                yield ModelLoadingResponse(layers_loaded=i, total=total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

        coupled_drafter, drafter_id, drafter_model = _try_load_collocated_drafter(
            target_card, model, allow_standard_drafter_fallback=True
        )

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = yield from shard_and_load(
            bound_instance.bound_shard,
            group=group,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

        # Symmetric multi-rank (tensor-parallel) placements reach the same
        # coupled-drafter loader as single-device: each rank replicates the
        # small coupled drafter and consumes the post-all-reduce hidden state
        # locally. Standard external drafters can't dispatch through a group,
        # so no fallback here.
        coupled_drafter, drafter_id, drafter_model = _try_load_collocated_drafter(
            target_card, model, allow_standard_drafter_fallback=False
        )

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from exo.worker.engines.mlx.vision import VisionProcessor

        vision_start_time = time.perf_counter()
        try:
            vision_processor: VisionProcessor | None = VisionProcessor(
                vision_config, bound_instance.bound_shard.model_card.model_id
            )
            vision_processor.load()
            logger.info(
                f"Time taken to load vision weights: {(time.perf_counter() - vision_start_time):.2f}s"
            )
        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to load vision weights — disabling vision for this runner"
            )
            vision_processor = None
    else:
        vision_processor = None

    return (
        cast(Model, model),
        tokenizer,
        vision_processor,
        drafter_model,
        drafter_id,
        coupled_drafter,
    )


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, tuple[nn.Module, TokenizerWrapper]]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    # Probe peers' TcpRelay ports to determine CUDA ranks (non-invasive).
    # TcpRelay servers are started eagerly during distributed init.
    import platform as _plat
    import socket as _sock
    import struct as _struct

    _is_cuda = (
        _plat.system() == "Linux" and mx.default_device().type == mx.DeviceType.gpu
    )
    _cuda_ranks = []
    if _is_cuda:
        hosts_json = os.environ.get("MLX_HOSTS_JSON", "[]")
        import json as _json

        _hosts = _json.loads(hosts_json)
        _my_rank = int(os.environ.get("MLX_RANK", "0"))
        for i, h in enumerate(_hosts):
            if i == _my_rank:
                if _is_cuda:
                    _cuda_ranks.append(str(i))
                continue
            ip = str(h).split(":")[0] if not isinstance(h, dict) else h.get("ip", "")
            peer_port = 40000 + i
            try:
                s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((ip, peer_port))
                s.setsockopt(
                    _sock.SOL_SOCKET, _sock.SO_LINGER, _struct.pack("ii", 1, 0)
                )
                s.close()
                _cuda_ranks.append(str(i))
            except Exception:
                pass
    os.environ["MLX_CUDA_RANKS"] = ",".join(_cuda_ranks)
    logger.info(f"CUDA ranks probed: {_cuda_ranks}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = yield from tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = yield from pipeline_auto_parallel(model, group, shard_metadata)
            mx.eval(model.parameters())
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # For GLM-4.7 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # For Qwen3.5 / Qwen3.6: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    # We should really consider going back to mlx lm load to get tokenizer
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def _needs_v4_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """Remove `<think>…</think>` blocks and any stray `<think>`/`</think>` tags
    from prior-turn assistant content.

    The V4 encoder drops `reasoning_content` for older turns when
    `drop_thinking=True`"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    System messages almost exclusively must go at the start of a message
    and there must only be a single one.

    Also, Codex sends "developer" messages which are just system prompts.
    """
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = cast(str, msg.get("content", ""))
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)
    formatted_messages = non_system
    if system_parts:
        formatted_messages.insert(
            0, {"role": "system", "content": "\n".join(system_parts)}
        )
    return formatted_messages


def render_chat_template(
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, Any]],
    task_params: TextGenerationTaskParams,
) -> str:
    """
    Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages = consolidate_system_messages(messages)

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # Only use chat mode if enable thinking is explicitly Fakse.
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    if _needs_v4_encoding(task_params):
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
            encode_messages as encode_messages_v4,
        )

        v4_messages = [dict(m) for m in formatted_messages]
        for msg in v4_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _strip_v4_thinking_markers(content)
        if task_params.tools:
            for msg in v4_messages:
                if msg.get("role") in ("system", "developer"):
                    msg["tools"] = task_params.tools
                    break
            else:
                v4_messages.insert(
                    0, {"role": "system", "content": "", "tools": task_params.tools}
                )

        prompt = encode_messages_v4(
            messages=v4_messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=_v4_reasoning_effort(task_params),
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    # Put reasoning content in thinking block for GPT OSS
    if "gpt-oss" in task_params.model.lower():
        for msg in formatted_messages:
            if msg.get("role") == "assistant" and "thinking" not in msg:
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    msg["thinking"] = rc

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    return prompt


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    messages: list[dict[str, ChatTemplateValue]] = []
    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        messages = task_params.chat_template_messages
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            messages.append({"role": "system", "content": task_params.instructions})

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            messages.append({"role": msg.role, "content": msg.content})

    prompt = render_chat_template(tokenizer, messages, task_params)
    logger.debug(prompt)

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """Approximate token count of the system prompt portion of the input."""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    assert tokenizer.think_start_tokens
    assert tokenizer.think_end_tokens
    think_start_tokens: list[int] = tokenizer.think_start_tokens
    think_end_tokens: list[int] = tokenizer.think_end_tokens
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []

    depth = 0
    accumulated_think_start_length = 0
    accumulated_think_end_length = 0

    for token in token_list:
        if token == think_start_tokens[accumulated_think_start_length]:
            accumulated_think_start_length += 1
            if accumulated_think_start_length == len(think_start_tokens):
                depth += 1
                accumulated_think_start_length = 0

        elif token == think_end_tokens[accumulated_think_end_length]:
            accumulated_think_end_length += 1
            if accumulated_think_end_length == len(think_end_tokens):
                if depth == 0:
                    result.extend(think_start_tokens)
                else:
                    depth -= 1
                accumulated_think_end_length = 0

        else:
            accumulated_think_start_length = 0
            accumulated_think_end_length = 0

        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory) -> None:
    if mx.metal.is_available():
        max_rec_size = Memory.from_bytes(
            int(mx.device_info()["max_recommended_working_set_size"])
        )
        if model_size > 0.9 * max_rec_size:
            logger.warning(
                f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
                f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        mx.set_wired_limit(max_rec_size.in_bytes)
        logger.info(f"Wired limit set to {max_rec_size}.")
    elif hasattr(mx, "cuda") and mx.cuda.is_available():
        logger.info("CUDA backend active — skipping Metal wired limit.")


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: mx.distributed.Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: mx.distributed.Group | None):
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(mx.array([n_tasks]), group=group).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = 1 if group is None else group.size()

    if max_tasks == 0:
        return [], []

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)

    assert all(len(encoded_task_id) == uuid_byte_length for encoded_task_id in padded)

    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids = set[TaskId].intersection(*(set(tids) for tids in all_task_ids))

    local_tasks = {task.task_id: task for task in tasks}
    agreed = [local_tasks[tid] for tid in sorted(agreed_ids)]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different


@final
@dataclass(frozen=True)
class TargetPeerFanout:
    """Direct TCP int-broadcast wire between target rank 0 and its peers.

    Replaces :func:`mx.distributed.send` / :func:`recv` on the
    spec-decode hot path. JACCL on Apple Silicon conflates int32
    broadcasts on the target group with the model's float32 TP
    ``all_sum`` collectives; the former occasionally returns the
    latter's logit memory reinterpreted as int32, surfacing as
    out-of-vocab token ids (~``10^9``) deep in the SPM detokenizer.

    The model's TP ``all_sum`` collectives stay on JACCL/RDMA -- they
    carry multi-MB tensor reductions where vendor RDMA wins
    decisively. Only the tiny (~24-byte) int32 broadcasts move to TCP,
    where Thunderbolt with ``TCP_NODELAY`` adds <100µs per round
    (negligible against a ~30ms verifier forward).

    Topology:
      * On target rank 0: ``peer_sockets`` holds one connection per
        non-zero peer rank, indexed by peer rank.
      * On a peer target rank (rank > 0): ``rank_zero_socket`` holds
        the single connection back to rank 0.

    Both shapes are produced by :func:`_setup_target_peer_fanout` at
    instance bootstrap and are immutable for the runner's lifetime.
    Reconnect-on-failure is intentionally NOT supported: a transport
    failure on this wire is treated as a hard runner failure (same as
    a TP all-reduce failure) and the supervisor rebuilds the instance.
    """

    rank: int
    """Caller's target rank inside the parent group; matches
    ``MlxGroupSplit.parent.rank()`` when ``parent`` is set."""

    peer_sockets: dict[int, object] = field(default_factory=dict)
    """Rank 0 only: ``{peer_rank: socket.socket}``. Empty on rank > 0."""

    rank_zero_socket: object | None = None
    """Rank > 0 only: connected socket back to rank 0. ``None`` on rank 0."""

    expected_world_size: int = 1
    """Target world size (every rank in the fanout sees the same value).

    Stored explicitly so the broadcast helpers can sanity-check that
    rank 0's ``peer_sockets`` cover all peers without re-deriving the
    world size from a possibly-discarded group handle."""


_MX_BROADCAST_MAX_VALUE: Final[int] = (1 << 31) - 1
# Toggle to dump every broadcast call's send/recv buffers. Set via
# ``EXO_PROBE_BROADCAST=1`` for ad-hoc diagnostics; leave off in
# steady state because the per-token logging spam quickly dominates.
_BROADCAST_PROBE: Final[bool] = bool(os.environ.get("EXO_PROBE_BROADCAST"))


# Distributed backend literal -- matches the strings we pass to
# ``mx.distributed.init(backend=...)`` in :func:`mlx_distributed_init`.
DistributedBackend = Literal["ring", "jaccl"]


def _detect_distributed_backend() -> DistributedBackend:
    """Resolve the active MLX distributed backend from the env vars
    set by :func:`mlx_distributed_init`.

    Why env-var sniffing instead of asking the group: ``mx.distributed.Group``
    only exposes ``rank()`` / ``size()`` / ``split()`` and gives no
    public hook for the backend name. We control the init path
    (:func:`mlx_distributed_init`) and set ``MLX_HOSTFILE`` for ring
    and ``MLX_IBV_DEVICES`` (plus ``MLX_JACCL_COORDINATOR``) for
    jaccl, so checking those env vars is a deterministic, in-process
    signal that doesn't require threading a backend literal through
    every call site.

    Backend selection matters because the ring backend is built around
    collective primitives (``all_sum`` / ``all_gather``) and does not
    support arbitrary point-to-point ``send`` / ``recv`` between
    non-neighbor ranks; multi-rank ring deployments would fail or
    hang the moment :func:`mx_broadcast_int_list` issued a
    ``send(dst=N)`` for a non-neighbor ``N``. JACCL, on the other
    hand, supports arbitrary ``send`` / ``recv`` and we deliberately
    use that to keep int32 broadcasts off the same all-reduce wire as
    TP float32 collectives (see the docstring on
    :func:`mx_broadcast_int_list` for the historical wire-conflation
    bug).

    Returns:
      ``"ring"`` when ``MLX_HOSTFILE`` is set, else ``"jaccl"``.
      Defaults to ``"ring"`` when neither marker is present so the
      ring-safe code path runs in ambiguous setups (e.g. tests that
      construct a fake group without going through
      :func:`mlx_distributed_init`).

    Raises:
      None. Detection is best-effort by design: the caller already
      gated multi-rank entry on ``group is not None``, and a
      misdetected backend at most picks the slower-but-correct
      collective path.
    """
    if os.environ.get("MLX_HOSTFILE"):
        return "ring"
    if os.environ.get("MLX_IBV_DEVICES") or os.environ.get("MLX_JACCL_COORDINATOR"):
        return "jaccl"
    return "ring"


def _validate_broadcast_values(values: list[int]) -> None:
    """Range-check root-side broadcast values.

    Centralised so both the single-rank short-circuit and the multi-
    rank all-sum path enforce identical contracts. Linear scan; for
    ``length`` values this is microseconds and runs once per round on
    the spec-decode hot path -- amortised free against an MLX
    collective.
    """
    for index, value in enumerate(values):
        if value < 0 or value > _MX_BROADCAST_MAX_VALUE:
            raise ValueError(
                f"mx_broadcast_int_list values must be in "
                f"[0, {_MX_BROADCAST_MAX_VALUE}]; "
                f"index {index} = {value} is out of range "
                f"(negatives wrap silently in int32 all-sum; values "
                f">= 2**31 overflow)"
            )


def mx_broadcast_int_list(
    values: list[int] | None,
    length: int,
    group: mx.distributed.Group | None,
    *,
    is_root: bool,
) -> list[int]:
    """Broadcast a fixed-length int list from one rank to all peers.

    Backend-aware implementation:

      * ``ring``: use ``all_sum`` of an int32 buffer where non-root
        ranks contribute zeros and root contributes ``values``. Sum
        across the group recovers ``values`` element-wise (root's
        contribution is the only nonzero summand). MLX's ring backend
        is built around collective primitives and does not support
        arbitrary point-to-point ``send`` / ``recv`` between
        non-neighbor ranks, so this is the only ring-safe option.
      * ``jaccl``: rank-0 fanout via :func:`mx.distributed.send` /
        :func:`mx.distributed.recv`. Root issues one send to every
        peer; each peer issues a single matching recv from rank 0.

    Why split by backend: under JACCL the model's TP layers issue
    ``all_sum`` on the same target group on float32 buffers, every
    layer, every forward. A previous revision used ``all_sum`` for
    this broadcast on JACCL too and observed silent corruption on
    the spec-decode hot path: with >100 in-flight ``all_sum``
    collectives per round all on the same group, JACCL's pairing
    logic occasionally matched our int32 "broadcast" on rank A
    against the model's float32 TP all-reduce on rank B, scrambling
    the int32 buffer (symptom: token ids ~10^9 emitted by the spec
    loop, ``IndexError`` deep in the SPM detokenizer). Switching to
    ``send`` / ``recv`` on JACCL makes this broadcast a different
    primitive than the TP all-reduce so JACCL has no opportunity to
    merge them. Ring lacks both the JACCL pairing pitfall and the
    arbitrary-``send`` capability, so it stays on ``all_sum``.

    Caller note: the spec-decode hot path no longer routes through
    this function -- it uses :func:`target_peer_broadcast_int_list`
    over a dedicated TCP fanout (see :class:`TargetPeerFanout`). The
    only remaining caller is :func:`mx_all_gather_tasks` at admit
    boundaries, which fires far below TP all-reduce frequency, so
    even on JACCL the wire-conflation risk is low; the
    ``send`` / ``recv`` path is kept for defense-in-depth.

    The fixed-length contract means the caller pads to ``length`` on
    root and both ranks agree on ``length`` ahead of time, which keeps
    the recv shape (or all_sum buffer shape) known statically.

    Args:
      values: On root, a list of exactly ``length`` ints to broadcast.
        Each value must be in ``[0, 2**31 - 1]``. Negative values are
        rejected explicitly so a stray ``-1`` doesn't silently wrap
        on the int32 cast and corrupt the broadcast. Ignored on
        non-root.
      length: Buffer size, agreed by all ranks. Must be ``>= 1``.
      group: Distributed group; ``None`` is a single-rank short-circuit
        that simply returns ``values`` (root-only).
      is_root: ``True`` on the rank holding the source values; ``False``
        elsewhere. Exactly one rank in ``group`` must pass ``True``.

    Returns:
      A list of ``length`` ints identical on every rank in ``group``,
      equal to root's ``values``.

    Raises:
      ValueError: ``length`` is non-positive, the root's ``values`` are
        ``None`` or wrong length, or any root value is out of int32
        range. These are caller bugs, not runtime conditions.
    """
    if length < 1:
        raise ValueError(f"mx_broadcast_int_list length must be >= 1, got {length}")

    if group is None:
        if not is_root:
            raise ValueError(
                "mx_broadcast_int_list: single-rank short-circuit requires "
                "is_root=True (only the root has source values)"
            )
        if values is None or len(values) != length:
            raise ValueError(
                "mx_broadcast_int_list: single-rank call requires "
                f"values of length {length}, got "
                f"{None if values is None else len(values)}"
            )
        _validate_broadcast_values(values)
        return list(values)

    group_size = group.size()

    if is_root and (values is None or len(values) != length):
        raise ValueError(
            "mx_broadcast_int_list root rank requires values of "
            f"length {length}, got {None if values is None else len(values)}"
        )
    if is_root:
        # ``cast`` for the type-checker: validated above.
        _validate_broadcast_values(cast(list[int], values))

    backend = _detect_distributed_backend()

    if backend == "ring":
        # Ring backend: collective ``all_sum``. Root contributes the
        # values, every other rank contributes a zero buffer of the
        # same shape, so the element-wise sum is ``values``. This is
        # the only ring-safe broadcast primitive (ring rejects
        # arbitrary point-to-point ``send`` / ``recv`` between
        # non-neighbor ranks).
        if is_root:
            local = mx.array(cast(list[int], values), dtype=mx.int32)
        else:
            local = mx.zeros(shape=(length,), dtype=mx.int32)
        summed = mx.distributed.all_sum(local, group=group)
        mx.eval(summed)
        out = [int(v) for v in cast(list[int], summed.tolist())]
        if _BROADCAST_PROBE:
            role = "ROOT" if is_root else "PEER"
            logger.warning(
                f"mx_broadcast_int_list[ring] {role} recovered {out} (len={length})"
            )
        return out

    # JACCL backend: send/recv fanout from rank 0.
    if is_root:
        send_buffer = mx.array(cast(list[int], values), dtype=mx.int32)
        for dst in range(1, group_size):
            sent = mx.distributed.send(send_buffer, dst=dst, group=group)
            mx.eval(sent)
        if _BROADCAST_PROBE:
            logger.warning(
                f"mx_broadcast_int_list[jaccl] ROOT sent {values} (len={length})"
            )
        return list(cast(list[int], values))

    received = mx.distributed.recv(shape=(length,), dtype=mx.int32, src=0, group=group)
    mx.eval(received)
    out = [int(v) for v in cast(list[int], received.tolist())]
    if _BROADCAST_PROBE:
        logger.warning(
            f"mx_broadcast_int_list[jaccl] PEER recvd {out} (expected len={length})"
        )
    return out


def target_peer_broadcast_int_list(
    values: list[int] | None,
    length: int,
    fanout: TargetPeerFanout,
    *,
    is_root: bool,
) -> list[int]:
    """Broadcast a fixed-length signed int list over the TCP fanout.

    Drop-in replacement for :func:`mx_broadcast_int_list` on the
    spec-decode hot path. Same shape contract (``length`` agreed by
    every rank up front; root passes ``values``, peers pass
    ``None``); the only difference is that this version rides direct
    TCP sockets instead of ``mx.distributed.send`` / ``recv``,
    sidestepping the JACCL int/float wire-conflation bug entirely.

    Wire format (every frame): ``length`` little-endian signed int32
    values, no header. The peer side knows ``length`` from the same
    shape contract the caller agreed to.

    Args:
      values: On root, exactly ``length`` int32-range values to
        broadcast. Ignored on peers.
      length: Buffer size, agreed by all ranks. Must be ``>= 1``.
      fanout: Pre-built fanout from :func:`_maybe_setup_target_peer_fanout`.
        Carries the per-rank role (rank 0 vs peer) and the connected
        sockets. Mismatched ``is_root`` vs ``fanout.rank`` is a caller
        bug and raises :class:`ValueError`.
      is_root: ``True`` on rank 0, ``False`` elsewhere. Asserted
        against ``fanout.rank``.

    Returns:
      A list of ``length`` ints identical on every rank, equal to
      root's ``values``.

    Raises:
      ValueError: caller-bug conditions (length, values shape,
        is_root vs rank mismatch).
      ConnectionError: a peer closed the socket mid-frame; surfaces
        as a runner failure for the supervisor to rebuild.
    """
    import socket as _socket

    from exo.worker.engines.mlx.generator.target_peer_socket import (
        recv_int32_frame,
        send_int32_frame,
    )

    if length < 1:
        raise ValueError(
            f"target_peer_broadcast_int_list length must be >= 1, got {length}"
        )
    if is_root != (fanout.rank == 0):
        raise ValueError(
            f"target_peer_broadcast_int_list is_root={is_root} disagrees "
            f"with fanout.rank={fanout.rank}; exactly one rank in the "
            "fanout must pass is_root=True"
        )
    if is_root:
        if values is None or len(values) != length:
            raise ValueError(
                "target_peer_broadcast_int_list root rank requires values "
                f"of length {length}, got "
                f"{None if values is None else len(values)}"
            )
        for sock in fanout.peer_sockets.values():
            assert isinstance(sock, _socket.socket)  # narrow object -> socket
            send_int32_frame(sock, values)
        return list(values)
    sock = fanout.rank_zero_socket
    if sock is None:
        raise RuntimeError(
            "target_peer_broadcast_int_list called on peer rank but "
            "fanout.rank_zero_socket is None; bootstrap must populate it"
        )
    assert isinstance(sock, _socket.socket)
    return recv_int32_frame(sock, length)


EXO_DISABLE_DRAFTER_ENV = "EXO_DISABLE_DRAFTER"
EXO_DRAFTER_PREFERENCE_ENV = "EXO_DRAFTER_PREFERENCE"

# Allowed values for ``EXO_DRAFTER_PREFERENCE``. ``fastest`` picks the first
# drafter declared on the card (smallest by convention); ``highest_acceptance``
# picks the last (largest by convention); ``auto`` defaults to ``fastest`` but
# may be tuned by future heuristics (e.g. observed acceptance rate).
_DRAFTER_PREFERENCE_VALUES: frozenset[str] = frozenset(
    {"fastest", "highest_acceptance", "auto"}
)


def _drafter_disabled_by_env() -> bool:
    return os.environ.get(EXO_DISABLE_DRAFTER_ENV, "").lower() in {"1", "true", "yes"}


def _drafter_preference() -> str:
    raw = os.environ.get(EXO_DRAFTER_PREFERENCE_ENV, "auto").lower()
    if raw not in _DRAFTER_PREFERENCE_VALUES:
        logger.warning(
            f"Unknown {EXO_DRAFTER_PREFERENCE_ENV}={raw!r}, falling back to 'auto'"
        )
        return "auto"
    return raw


# Drafter kinds the loader recognises. ``"standard"`` is the existing
# external-drafter path (independent sibling LM via mlx-lm). ``"mtp"`` and
# ``"dflash"`` are the coupled-drafter kinds shipped by mlx-vlm 0.5+ that
# attach to the target architecturally (consume the target's hidden state /
# KV cache every draft step) and only run on single-node placements.
CoupledDrafterKind = Literal["mtp", "dflash"]
_KNOWN_COUPLED_DRAFTER_KINDS: Final[frozenset[CoupledDrafterKind]] = frozenset(
    {"mtp", "dflash"}
)


@final
@dataclass(frozen=True, kw_only=True)
class CoupledDrafter:
    """A loaded MTP/DFlash-kind coupled drafter, ready for the generator.

    Coupled drafters consume the target's hidden state every draft step and
    (for ``kind="mtp"``) read the target's KV cache directly via
    ``set_shared_kv``. They cannot decode independently the way standard
    external drafters can, so this loader path runs only when the placement
    collocates target + drafter on the same node (i.e. the target is not
    asymmetrically split via ``DrafterPlacement`` and the runner is loading
    both halves locally).

    The model object is typed ``object`` because the concrete class
    (``Gemma4AssistantDraftModel`` for ``mtp``, ``DFlashDraftModel`` for
    ``dflash``) lives in mlx-vlm and importing it in the worker hot path
    would force every linux/CPU build to drag mlx-vlm into the type
    surface. Generator-side dispatch narrows the type at the use site.
    """

    model_id: ModelId
    kind: CoupledDrafterKind
    model: object


# Exceptions :func:`_dispatch_attach_coupled_hooks` may raise that the
# loader caller should treat as "drafter loaded but not dispatchable on
# this target -- degrade to standard drafting" rather than crashes:
#
# - :class:`TypeError` -- right kind, wrong target architecture (e.g.
#   card declared a ``coupled_drafter`` of kind ``"mtp"`` but the target
#   loaded as something other than a Gemma 4 ``Model``).
# - :class:`exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks.DFlashHooksNotImplementedError`
#   -- right kind, hooks not yet vendored for that kind. Today raised by
#   the dflash skeleton; deletion follows the qwen3_5 vendor work.
#
# Listed at module scope (rather than caught inline) so the exception
# tuple stays a single source of truth -- adding a future coupled-drafter
# kind extends the tuple here once and the loader picks it up automatically.
# ``_DFlashHooksNotImplementedError`` is imported at the top of the file
# alongside other vendor imports so ruff E402 stays happy.
_COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = (
    TypeError,
    _DFlashHooksNotImplementedError,
)


def _dispatch_attach_coupled_hooks(kind: CoupledDrafterKind, model: object) -> None:
    """Mark ``model`` as wired for ``kind``'s coupled-drafter hooks.

    Per-kind dispatcher around the vendor modules' ``attach_*_hooks``
    helpers. Splitting the dispatch out of the load path lets the
    loader stay kind-agnostic -- adding a new coupled-drafter kind
    only requires extending this match plus the vendor module, not
    touching :func:`load_mlx_items`.

    Raises:
        TypeError: ``model`` is the wrong target architecture for
            the declared ``kind``. Caller falls back to standard
            drafting (see :data:`_COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS`).
        DFlashHooksNotImplementedError: ``kind == "dflash"`` and the
            qwen3_5 hook surface is still a skeleton. Same fallback.
    """
    match kind:
        case "mtp":
            from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
                attach_mtp_hooks,
            )

            attach_mtp_hooks(model)
        case "dflash":
            from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
                attach_dflash_hooks,
            )

            attach_dflash_hooks(model)


def _coupled_drafter_weight_size_bytes(coupled_id: ModelId) -> int:
    """Best-effort coupled-drafter on-disk size for the wired-memory bump.

    Mirrors :func:`_drafter_weight_size_bytes`: walk the drafter directory
    and sum file sizes; return 0 on any error. Coupled drafters are tiny
    (~158MB for the Gemma 4 E2B assistant) so under-wiring here is cheap
    even if the helper falls through; we just want a reasonable hint to
    ``set_wired_limit_for_model`` so the OS doesn't page the drafter
    weights out between requests.
    """
    drafter_path = resolve_existing_model(coupled_id)
    if drafter_path is None:
        return 0
    try:
        return sum(p.stat().st_size for p in drafter_path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _try_load_coupled_drafter(model_card: ModelCard) -> CoupledDrafter | None:
    """Attempt to load the coupled drafter declared on ``model_card``.

    Returns the loaded drafter on success, or ``None`` when:
    - the card declares no ``coupled_drafter``,
    - ``EXO_DISABLE_DRAFTER`` is set,
    - mlx-vlm is unavailable (e.g. linux build without the speculative
      drafters extra) or too old to expose ``load_drafter``,
    - the drafter's weights are not on disk,
    - mlx-vlm resolves an unknown / unsupported drafter kind, or
    - the load itself raises.

    Failures are logged at warning level and swallowed so that single-node
    deployments degrade to the standard external-drafter list (or to plain
    decoding) instead of crashing the runner. The caller is responsible
    for that fallback.
    """
    coupled_id = model_card.coupled_drafter
    if coupled_id is None:
        return None
    if _drafter_disabled_by_env():
        logger.info(
            f"Coupled drafter declared by {model_card.model_id} but "
            f"{EXO_DISABLE_DRAFTER_ENV} is set; skipping coupled drafter load."
        )
        return None

    # mlx-vlm's speculative-drafter API is partially typed (its
    # ``load_drafter`` signature uses ``**kwargs`` with no annotation),
    # so we cast at the import boundary to give the rest of this
    # function a well-typed surface. ``KNOWN_DRAFTER_KINDS`` is an
    # iterable of upstream kind strings -- declared as ``Iterable[str]``
    # because mlx-vlm uses ``frozenset[str]`` today but a future
    # release could swap it for a list without breaking us.
    #
    # Codex P2 (PR #23 round-(N+0), utils_mlx.py:809): we also catch
    # ``AttributeError`` so a partial / mismatched mlx-vlm install (the
    # ``speculative`` package imports cleanly but is missing
    # ``load_drafter`` / ``KNOWN_DRAFTER_KINDS`` -- e.g. an old release
    # with the namespace package but pre-drafter API, or a future
    # release that renames the symbols) degrades to the standard
    # drafter path instead of crashing the runner.
    try:
        from mlx_vlm.speculative import (  # pyright: ignore[reportMissingTypeStubs]
            drafters as _mlxvlm_drafters,
        )

        load_drafter = cast(
            Callable[..., tuple[object, str]],
            _mlxvlm_drafters.load_drafter,
        )
        known_drafter_kinds = cast(
            "Iterable[str]",
            _mlxvlm_drafters.KNOWN_DRAFTER_KINDS,
        )
    except (ImportError, AttributeError) as exc:
        logger.warning(
            f"Coupled drafter declared by {model_card.model_id} requires "
            f"mlx-vlm with speculative-drafter support (>=0.5.0) exposing "
            f"``load_drafter`` and ``KNOWN_DRAFTER_KINDS``, but resolving "
            f"those symbols failed ({type(exc).__name__}: {exc}); falling "
            f"back to the standard drafter path."
        )
        return None

    drafter_path = resolve_existing_model(coupled_id)
    if drafter_path is None:
        logger.warning(
            f"Coupled drafter {coupled_id} declared by {model_card.model_id} "
            "is not downloaded; pre-download it to enable coupled "
            "speculative decoding. Falling back to the standard drafter "
            "path for this load."
        )
        return None

    drafter_start = time.perf_counter()
    try:
        loaded_model, resolved_kind = load_drafter(str(drafter_path), kind=None)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            f"Failed to load coupled drafter {coupled_id} via mlx-vlm; "
            "falling back to the standard drafter path."
        )
        return None

    if resolved_kind not in _KNOWN_COUPLED_DRAFTER_KINDS:
        # mlx-vlm may evolve to recognise more kinds before exo's loader
        # learns to dispatch them; refuse rather than load a model the
        # generator cannot drive.
        known_upstream: list[str] = sorted(known_drafter_kinds)
        logger.warning(
            f"Coupled drafter {coupled_id} resolved to kind "
            f"{resolved_kind!r}, which exo's generator does not yet "
            f"support (known kinds: {sorted(_KNOWN_COUPLED_DRAFTER_KINDS)}; "
            f"mlx-vlm reports: {known_upstream}). Falling "
            "back to the standard drafter path."
        )
        return None

    logger.info(
        f"Loaded coupled drafter {coupled_id} (kind={resolved_kind!r}) "
        f"for {model_card.model_id} in "
        f"{(time.perf_counter() - drafter_start):.2f}s"
    )
    return CoupledDrafter(
        model_id=coupled_id,
        kind=resolved_kind,
        model=loaded_model,
    )


def _select_drafter_id(candidates: list[ModelId], preference: str) -> ModelId | None:
    """Pick a drafter id from a card's preference-ordered list.

    The card lists drafters in `[fastest, ..., highest_acceptance]` order. We
    prefer drafters that are already on disk (so the chooser doesn't force a
    surprise download); within the on-disk subset we honor the user's
    preference. If nothing is on disk we fall back to the head of the list,
    leaving the loader to log a "weights missing" warning.
    """
    if not candidates:
        return None

    on_disk = [cid for cid in candidates if resolve_existing_model(cid) is not None]
    pool = on_disk if on_disk else candidates

    if preference == "highest_acceptance":
        return pool[-1]
    return pool[0]


def _maybe_load_drafter(model_card: ModelCard) -> tuple[ModelId, Model] | None:
    """Load a drafter model declared on ``model_card``, if any.

    Returns the chosen ``(drafter_id, drafter_model)`` pair on success, or
    ``None`` when the card declares no drafter, the chosen drafter's weights
    are not on disk, ``EXO_DISABLE_DRAFTER`` is set, or the load itself
    fails. Drafter loading failures are logged and swallowed: the target
    model continues to load and inference falls back to standard
    (non-speculative) decoding.

    This helper is intentionally single-device only. Multi-device distributed
    inference does not pass ``draft_model`` through to ``stream_generate``
    today (see ``mlx_generate``), so loading a drafter on those ranks would
    just waste memory.
    """
    candidates = list(model_card.drafter_model_ids)
    if not candidates:
        return None
    if _drafter_disabled_by_env():
        logger.info(
            f"Drafter declared by {model_card.model_id} but "
            f"{EXO_DISABLE_DRAFTER_ENV} is set; skipping drafter load."
        )
        return None

    preference = _drafter_preference()
    drafter_id = _select_drafter_id(candidates, preference)
    if drafter_id is None:
        return None

    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        logger.warning(
            f"Drafter {drafter_id} (preferred '{preference}') declared by "
            f"{model_card.model_id} is not downloaded; falling back to "
            "standard decoding. Pre-download the drafter to enable "
            "speculative decoding."
        )
        return None

    drafter_start = time.perf_counter()
    try:
        drafter_model, _ = load_model(drafter_path, lazy=True, strict=False)
        mx.eval(drafter_model)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            f"Failed to load drafter {drafter_id}; continuing without "
            "speculative decoding."
        )
        return None
    logger.info(
        f"Loaded drafter {drafter_id} (preferred '{preference}') for "
        f"{model_card.model_id} in {(time.perf_counter() - drafter_start):.2f}s"
    )
    return drafter_id, cast(Model, drafter_model)


def _try_load_collocated_drafter(
    target_card: ModelCard,
    model: nn.Module,
    *,
    allow_standard_drafter_fallback: bool,
) -> tuple[CoupledDrafter | None, ModelId | None, Model | None]:
    """Resolve the collocated drafter (coupled or standard) for ``model``.

    Coupled-drafter precedence: when the card declares
    ``coupled_drafter`` we try it first because it's the path that
    yields the multi-x DFlash / MTP speedup. If the coupled load
    fails (mlx-vlm missing, weights absent, kind unrecognised, target
    type unsupported) we either fall through to the standard
    external-drafter list (single-device, where the standard drafter
    *is* dispatchable) or return empty-handed (multi-device, where
    the generator can't dispatch standard drafters yet so loading
    one would just waste memory).

    On a successful coupled load we ALSO attach the target-side hooks
    (``attach_mtp_hooks`` / ``attach_dflash_hooks``). The hook is the
    *capability gate* that :func:`mlx_generate` reads -- without it,
    the dispatch declines to route the request through the coupled
    path and the loaded coupled drafter stays passive. Hook
    attachment can fail on its own (e.g. the card incorrectly pairs a
    Gemma 4 ``coupled_drafter`` with a non-Gemma target); we treat
    that as another degrade-to-standard signal rather than a hard
    load failure so traffic keeps flowing through whichever drafter
    path is available.

    Used by both single-device and symmetric multi-rank (tensor-
    parallel) placements. Tensor parallel works because coupled
    drafters (~0.5-3 GB) replicate per rank and consume the post-
    all-reduce hidden state, which is identical on every rank. The
    drafter's own KV / SSM state replicates with the same logic.
    Asymmetric multi-rank uses a separate ``DrafterRunner`` reachable
    over the parent group and is handled by the caller (the
    ``drafter_placement is not None`` branch).

    Args:
        target_card: The target model card; supplies the
            ``coupled_drafter`` and ``drafter_model_ids`` declarations.
        model: The (possibly sharded) loaded target. Coupled hooks
            attach to this object's wrapper / inner-text-model
            sentinel attributes.
        allow_standard_drafter_fallback: Whether to fall back to
            :func:`_maybe_load_drafter` when no coupled drafter loads.
            Pass ``True`` for single-device placements (the standard
            drafter is dispatchable). Pass ``False`` for multi-device
            placements -- :func:`mlx_generate` declines to dispatch
            standard drafters when ``group is not None`` today, so a
            loaded standard drafter would just sit in memory unused.

    Returns:
        ``(coupled_drafter, drafter_id, drafter_model)`` where at
        most one of ``coupled_drafter`` and ``drafter_model`` is
        non-None. ``drafter_id`` is populated only on a successful
        standard-drafter load -- coupled-drafter attribution is
        threaded through ``GenerationStats`` from
        :data:`CoupledDrafter.model_id` instead, see
        :func:`_resolve_coupled_drafter_telemetry`.
    """
    coupled_drafter = _try_load_coupled_drafter(target_card)
    if coupled_drafter is not None:
        try:
            _dispatch_attach_coupled_hooks(coupled_drafter.kind, model)
        except _COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS as e:
            logger.warning(
                f"Coupled drafter loaded for "
                f"{target_card.model_id} but target type "
                f"{type(model).__name__!r} is incompatible "
                f"with the {coupled_drafter.kind} hooks "
                f"(error: {e}). Discarding coupled drafter "
                "and falling back to standard drafting."
            )
            coupled_drafter = None
    if coupled_drafter is not None:
        return coupled_drafter, None, None
    if not allow_standard_drafter_fallback:
        return None, None, None
    drafter_pair = _maybe_load_drafter(target_card)
    if drafter_pair is None:
        return None, None, None
    drafter_id, drafter_model = drafter_pair
    return None, drafter_id, drafter_model


def _drafter_weight_size_bytes(drafter_id: ModelId) -> int:
    """Best-effort drafter-on-disk size for the wired-memory bump.

    Walks the drafter directory and sums file sizes. Returns 0 on any error
    (the drafter weights aren't critical-path so we'd rather under-wire than
    crash).
    """
    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        return 0
    try:
        return sum(p.stat().st_size for p in drafter_path.rglob("*") if p.is_file())
    except OSError:
        return 0
