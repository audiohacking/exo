#!/usr/bin/env python3
"""Set up a disaggregated prefill/decode pair: two pinned single-node instances of
the same model (one on the prefill node, one on the decode node), linked via
/v1/instance-links so decode pulls prefill work from the linked node.

The dashboard UI only exposes a single multi-node instance with Tensor/Pipeline
sharding — neither of those routes prefill and decode to different hardware. This
script drives the actual disaggregated-prefill API flow instead, and is safe to
re-run: it reuses any existing instance/link rather than creating duplicates.

Assumes the model is already downloaded on both nodes. Requires
ENABLE_DISAGGREGATION=true on both nodes (checked via /v1/feature-flags first, with
a clear error if it isn't set) and, for the streaming KV path, EXO_STREAMING_PREFILL=1
on the prefill node specifically (not checked here — that's a startup-time env var
with no runtime-visible flag).

Usage:
    uv run python scripts/setup_disaggregated_cluster.py \\
        --model mlx-community/Qwen3.6-35B-A3B-bf16 \\
        --prefill-host spark-ams01:52415 \\
        --decode-host moysas-mac-studio:52415 \\
        --test
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

import httpx


class SetupError(RuntimeError):
    pass


async def get_json(
    client: httpx.AsyncClient,
    base: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    r = await client.get(f"{base}{path}", params=params)
    r.raise_for_status()
    return r.json()


async def post_json(
    client: httpx.AsyncClient, base: str, path: str, body: dict[str, Any]
) -> Any:
    r = await client.post(f"{base}{path}", json=body)
    r.raise_for_status()
    return r.json()


async def check_disaggregation_enabled(client: httpx.AsyncClient, api: str) -> None:
    flags = await get_json(client, api, "/v1/feature-flags")
    if not flags.get("disaggregation"):
        raise SetupError(
            f"ENABLE_DISAGGREGATION is not enabled on {api}. "
            "Set ENABLE_DISAGGREGATION=true on BOTH nodes and restart them, then retry."
        )


async def get_node_id(client: httpx.AsyncClient, host_base: str) -> str:
    r = await client.get(f"{host_base}/node_id")
    r.raise_for_status()
    return r.json()


async def find_existing_instance(
    client: httpx.AsyncClient, api: str, model_id: str, node_id: str
) -> str | None:
    state = await get_json(client, api, "/state")
    for instance_id, instance in state.get("instances", {}).items():
        inner = next(iter(instance.values()))
        shard_assignments = inner.get("shardAssignments", {})
        if shard_assignments.get("modelId") != model_id:
            continue
        if node_id in shard_assignments.get("nodeToRunner", {}):
            return instance_id
    return None


def pick_preview(previews_response: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        p
        for p in previews_response.get("previews", [])
        if p.get("error") is None and p.get("instance") is not None
    ]
    if not candidates:
        errors = {
            (p.get("sharding"), p.get("instance_meta")): p.get("error")
            for p in previews_response.get("previews", [])
        }
        raise SetupError(f"No valid single-node placement found. Errors seen: {errors}")
    preferred = [
        p
        for p in candidates
        if p.get("sharding") == "Pipeline" and p.get("instance_meta") == "MlxRing"
    ]
    return preferred[0] if preferred else candidates[0]


async def create_pinned_instance(
    client: httpx.AsyncClient, api: str, model_id: str, node_id: str
) -> str:
    previews = await get_json(
        client,
        api,
        "/instance/previews",
        params={"model_id": model_id, "node_ids": node_id},
    )
    preview = pick_preview(previews)
    resp = await post_json(client, api, "/instance", {"instance": preview["instance"]})

    # /instance doesn't return the new instance_id, only a command_id — poll /state
    # until it shows up (this normally takes well under a second).
    for _ in range(20):
        instance_id = await find_existing_instance(client, api, model_id, node_id)
        if instance_id:
            return instance_id
        await asyncio.sleep(0.5)
    raise SetupError(
        f"POST /instance succeeded (command_id={resp.get('command_id')}) but the "
        f"instance never appeared in /state for node={node_id} within 10s."
    )


async def runner_status_for(
    client: httpx.AsyncClient, api: str, instance_id: str, node_id: str
) -> dict[str, Any] | None:
    """Returns None if the instance or its runner hasn't shown up in /state yet —
    there's a brief window right after POST /instance where the instance exists but
    its runner entry doesn't, so callers should treat None as "still starting"."""
    state = await get_json(client, api, "/state")
    instance = state["instances"].get(instance_id)
    if instance is None:
        return None
    inner = next(iter(instance.values()))
    runner_id = inner["shardAssignments"]["nodeToRunner"].get(node_id)
    if runner_id is None:
        return None
    return state["runners"].get(runner_id)


async def wait_for_ready(
    client: httpx.AsyncClient,
    api: str,
    instance_id: str,
    node_id: str,
    label: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_kind: str | None = None
    while time.monotonic() < deadline:
        status = await runner_status_for(client, api, instance_id, node_id)
        kind = (
            "(runner not registered yet)"
            if status is None
            else next(iter(status.keys()))
        )
        if kind != last_kind:
            print(f"  [{label}] runner status: {kind}")
            last_kind = kind
        if status is None:
            await asyncio.sleep(2.0)
            continue
        if kind == "RunnerFailed":
            raise SetupError(f"[{label}] runner failed: {status[kind]}")
        if kind in ("RunnerReady", "RunnerRunning"):
            return
        await asyncio.sleep(2.0)
    raise SetupError(
        f"[{label}] timed out after {timeout_s:.0f}s waiting for ready "
        f"(last status: {last_kind})"
    )


async def ensure_link(
    client: httpx.AsyncClient, api: str, prefill_id: str, decode_id: str
) -> None:
    links = await get_json(client, api, "/v1/instance-links")
    for link in links:
        if prefill_id in link.get("prefillInstances", []) and decode_id in link.get(
            "decodeInstances", []
        ):
            print(f"  Link already exists: {link['linkId']}")
            return
    resp = await post_json(
        client,
        api,
        "/v1/instance-links",
        {"prefill_instances": [prefill_id], "decode_instances": [decode_id]},
    )
    print(f"  Created link (command_id={resp.get('command_id')})")


# The decode node only calls out to the linked prefill node if the uncached prompt
# exceeds EXO_REMOTE_PREFILL_MIN_TOKENS (default 1000, set on the decode node). A short
# test prompt "succeeding" only proves instance creation/linking works — it never
# touches the prefill node at all. This filler is long enough to comfortably clear the
# default threshold so --test actually exercises the remote-prefill data path.
_LONG_TEST_PROMPT = (
    "Summarize the key engineering tradeoffs of disaggregating LLM inference into "
    "separate prefill and decode stages across heterogeneous hardware. "
) * 60


async def test_chat_completion(
    client: httpx.AsyncClient, api: str, model_id: str
) -> None:
    print(
        "\nSending test chat completion (long prompt, to actually cross "
        "EXO_REMOTE_PREFILL_MIN_TOKENS and exercise the remote-prefill path — "
        "a short prompt would silently run entirely on the decode node)..."
    )
    async with client.stream(
        "POST",
        f"{api}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": _LONG_TEST_PROMPT}],
            "stream": True,
        },
        timeout=120.0,
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                print(line, flush=True)


async def run(args: argparse.Namespace) -> None:
    prefill_host = f"http://{args.prefill_host}"
    decode_host = f"http://{args.decode_host}"
    api = f"http://{args.api}" if args.api else prefill_host

    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"Checking ENABLE_DISAGGREGATION on {api} ...")
        await check_disaggregation_enabled(client, api)

        print("Resolving node IDs...")
        node_prefill = await get_node_id(client, prefill_host)
        node_decode = await get_node_id(client, decode_host)
        print(f"  prefill node ({args.prefill_host}) = {node_prefill}")
        print(f"  decode node  ({args.decode_host})  = {node_decode}")
        if node_prefill == node_decode:
            raise SetupError(
                "--prefill-host and --decode-host resolved to the SAME node ID — "
                "check your hostnames."
            )

        print(f"\nEnsuring prefill instance of {args.model} on {node_prefill}...")
        existing = await find_existing_instance(client, api, args.model, node_prefill)
        prefill_id = existing or await create_pinned_instance(
            client, api, args.model, node_prefill
        )
        print(
            f"  instance_id = {prefill_id}"
            + (" (reused)" if existing else " (created)")
        )

        print(f"\nEnsuring decode instance of {args.model} on {node_decode}...")
        existing = await find_existing_instance(client, api, args.model, node_decode)
        decode_id = existing or await create_pinned_instance(
            client, api, args.model, node_decode
        )
        print(
            f"  instance_id = {decode_id}" + (" (reused)" if existing else " (created)")
        )

        print("\nWaiting for prefill instance to be ready...")
        await wait_for_ready(
            client, api, prefill_id, node_prefill, "prefill", timeout_s=args.timeout
        )
        print("\nWaiting for decode instance to be ready...")
        await wait_for_ready(
            client, api, decode_id, node_decode, "decode", timeout_s=args.timeout
        )

        print(f"\nLinking prefill={prefill_id} -> decode={decode_id}...")
        await ensure_link(client, api, prefill_id, decode_id)

        print("\nDisaggregated cluster ready.")
        print(
            f"  prefill instance: {prefill_id} on {node_prefill} ({args.prefill_host})"
        )
        print(f"  decode  instance: {decode_id} on {node_decode} ({args.decode_host})")
        print(
            "  Verify streaming is active by tailing the prefill node's logs for "
            "'Streaming prefill:' lines during the next request."
        )

        if args.test:
            await test_chat_completion(client, api, args.model)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Full model_id, e.g. mlx-community/Qwen3.6-35B-A3B-bf16",
    )
    parser.add_argument(
        "--prefill-host",
        required=True,
        help="host:port of the prefill node's OWN API, e.g. spark-ams01:52415",
    )
    parser.add_argument(
        "--decode-host",
        required=True,
        help="host:port of the decode node's OWN API, e.g. moysas-mac-studio:52415",
    )
    parser.add_argument(
        "--api",
        default=None,
        help="Cluster API to drive placement/linking through (defaults to --prefill-host; "
        "any node's API works once the cluster is connected)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Seconds to wait for each instance to become ready (default 3600 — "
        "large models can take a while to load; this assumes the model is already "
        "downloaded on both nodes)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Send a test chat completion once linked"
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except SetupError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as e:
        print(
            f"\nHTTP ERROR: {e.request.method} {e.request.url} -> "
            f"{e.response.status_code} {e.response.text[:300]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
