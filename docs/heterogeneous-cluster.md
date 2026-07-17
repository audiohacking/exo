# Heterogeneous Cluster: Mac Studio + NVIDIA DGX Spark

Combine a Mac Studio (Metal GPU) and an NVIDIA DGX Spark (CUDA GPU) into a single exo cluster for **2.8× faster LLM inference** than either device alone.

## Architecture

exo uses **pipeline parallelism** with layer-by-layer KV cache streaming:

```
DGX Spark (compute-bound prefill)  →  KV cache stream  →  Mac Studio (memory-bound decode)
     ~100 TFLOPs FP16                              819 GB/s bandwidth
```

- **Prefill** (processing the prompt) is compute-bound → runs on the DGX Spark
- **Decode** (generating tokens one-by-one) is memory-bound → runs on the Mac Studio
- KV cache is streamed **layer-by-layer**, overlapping communication with computation to hide network latency
- A **10 GbE** (or faster) direct or switched connection between the two machines is essential — the KV cache transfer rate is the interconnect bandwidth

## Benchmarks

| Config | Total Time | Speedup |
|--------|-----------|---------|
| DGX Spark alone | 4.34s | 1.9× |
| Mac Studio M3 Ultra alone | 6.42s | 1.0× (baseline) |
| **DGX Spark + Mac Studio** | **2.32s** | **2.8×** |

*Source: Llama-3.1 8B, 8K context — [blog.exolabs.net/nvidia-dgx-spark](https://blog.exolabs.net/nvidia-dgx-spark/)*

---

## Prerequisites

### Hardware

| Device | Specs |
|--------|-------|
| **Mac Studio** | M3 Ultra chip, macOS 15.x+, Ethernet or Thunderbolt networking |
| **NVIDIA DGX Spark** | 128 GB RAM, ARM64 (aarch64), NVIDIA GPU with CUDA 13.0 support |
| **Network** | Direct 10 GbE or switched 10 GbE between the two machines. **This is critical.** The KV cache is streamed layer-by-layer between nodes during inference — any bottleneck on the interconnect directly limits end-to-end throughput. Wi-Fi or 1 GbE will not produce the expected results. |

### Software on Mac Studio

- macOS 15.x+
- [Xcode](https://developer.apple.com/xcode/) (provides the Metal ToolChain)
- [Homebrew](https://brew.sh/): `brew install uv node`
- [Rust](https://rustup.rs/): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && rustup toolchain install nightly`
- [macmon](https://github.com/vladkens/macmon) (pinned fork, required on Apple M5):
  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon --force
  ```

### Software on DGX Spark

- Ubuntu 24.04
- NVIDIA driver with CUDA 13.0 support
- Docker and Docker Compose

### Namespace

Both nodes must share the same `--namespace` (default is the exo version string `0.3.70`). Nodes with different namespaces will not discover each other.

---

## 1. Setting Up the Mac Studio Node

Clone our fork and build the dashboard:

```bash
git clone https://github.com/audiohacking/exo.git
cd exo

# Build the dashboard (one-time)
cd dashboard && npm install && npm run build && cd ..

# Run exo
uv run exo
```

**What starts:**

| Component | Port | Purpose |
|-----------|------|---------|
| API server | 52415 | OpenAI-compatible chat completions + dashboard |
| Zenoh router | 52414 | Peer-to-peer messaging |
| mDNS discovery | 52413 (UDP) | Automatic node discovery |

The node detects its backends as `[MlxCpu, MlxMetal]`.

**Verify:** Open `http://localhost:52415/` — the dashboard should show the node with GPU status.

**Optional flags:**

- `--force-master` — force this node to be the cluster master
- `--namespace custom-cluster` — custom namespace for cluster isolation

---

## 2. Setting Up the DGX Spark Node

### Clone our fork

On the DGX Spark, clone our fork (not the upstream repo) and switch to the branch with CUDA support:

```bash
git clone https://github.com/audiohacking/exo.git
cd exo
git checkout feature/linux-cuda-support
```

### Build the Docker image

Build the `exo:cuda13` image from our local code:

```bash
docker compose build
```

The multi-stage `docker/Dockerfile` builds from our source:

1. **Rust builder** (`nvidia/cuda:13.0.2-devel-ubuntu24.04`) — installs Rust nightly + maturin, copies `Cargo.toml`, `Cargo.lock`, and Rust source, then builds the `exo_rs` PyO3 wheel
2. **Dashboard builder** (`node:22-slim`) — copies `dashboard/package.json` and `dashboard/`, runs `npm ci` + `npm run build`
3. **Runtime** (`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`) — installs Python 3.13 + uv, copies `pyproject.toml`, `uv.lock`, `src/`, and `resources/`, creates a venv, runs `uv sync --extra mlx-cuda13 --no-install-project --no-install-workspace`, installs the prebuilt Rust wheel, then installs the Python package with `uv pip install . --no-deps`. Also ships an MLX CUDA compat shim (`mlx_cuda_compat.py`) that maps `mx.new_stream` to `mx.new_thread_local_stream` for MLX-LM compatibility

Key dependencies installed:

| Package | Version | Purpose |
|---------|---------|---------|
| `mlx-cuda-13` | 0.32.0 | MLX CUDA backend for Linux ARM64 |
| `mlx` | 0.32.0 | MLX core (from custom fork with JACCL fixes) |
| `numpy` | (latest) | TcpRelay tensor serialization |
| `torch` | 2.10.0 (cu130) | PyTorch CUDA 13.0 backend |

### Run the container

```bash
docker compose up
```

The `docker-compose.yml` config:

| Setting | Purpose |
|---------|---------|
| `network_mode: host` | Shares the host network so zenoh discovery and TcpRelay ports work directly |
| `deploy.resources.reservations.devices` | Exposes the GPU via the NVIDIA Container Toolkit |
| Volume mounts | Maps `~/.local/share/exo`, `~/.cache/exo`, `~/.cache/huggingface` into the container (NOT `~/.config/exo` — avoids sharing the Mac's keypair/identity) |
| `EXO_ZENOH_NAMESPACE` | Cluster namespace — must match the Mac node's namespace (e.g., `qxip`) |
| `EXO_ZENOH_PEERS` | Explicit peer addresses for non-multicast networks like Tailscale (e.g., `moysas-mac-studio:52414`) |

**Note:** When machines are on the same physical network, mDNS discovery works automatically and `EXO_ZENOH_PEERS` is not needed. Use it only when discovery fails (e.g., across Tailscale, which doesn't forward multicast).

**Verify:** Check the logs for:

```
CUDA check: os=Linux, device=..., is_linux_gpu=True
CUDA TcpRelay server started on port 40000
```

The node detects its backends as `[MlxCpu, MlxCuda, Vllm]`.

---

## 3. Connecting the Nodes

Discovery is **automatic** — no manual IP configuration needed. Both nodes broadcast on the same network using zenoh + mDNS on port `52413`.

### Verify discovery

From either node:

```bash
curl http://localhost:52415/state | jq '.nodes'
```

Expected output shows 2 nodes, one with `MlxMetal` backends and one with `MlxCuda` backends.

### Check the dashboard

Open `http://localhost:52415/` on either node. The cluster view should show both nodes connected.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| Nodes don't discover each other (same network) | Verify both are on the same subnet; check firewall allows UDP 52413 and TCP 52414 |
| Nodes don't discover each other (Tailscale) | Tailscale doesn't forward multicast. Set `EXO_ZENOH_PEERS=mac-hostname:52414` in the container's environment |
| Namespace mismatch | Check startup logs for `EXO_ZENOH_NAMESPACE` — both must match exactly. Set `EXO_ZENOH_NAMESPACE=qxip` (or your namespace) on both nodes |
| DGX Spark started first | It may have elected itself master. This is fine — the Mac will connect as a worker. Use `--force-master` on the Mac if you want it to be master. |
| Docker not finding GPU | Verify `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi` works |

---

## 4. Running Disaggregated Prefill/Decode (Recommended)

The Architecture section above describes DGX-does-prefill / Mac-does-decode with
layer-by-layer KV streaming. **The single multi-node instance described in section 5
below does not actually produce that architecture** — Tensor and Pipeline sharding both
run every prefill *and* decode step through both nodes' layer ranges. The real
DGX-prefill/Mac-decode architecture is a separate feature called **Instance Links**
(prefill/decode disaggregation): two independent single-node instances of the same
model, linked so the decode node pulls prefill work from the prefill node over TCP,
streaming KV cache out layer-by-layer as it's computed (`src/exo/worker/engines/mlx/disaggregated/streaming_prefill.py`).

The dashboard's model-launch dialog only exposes the Tensor/Pipeline choice — there is
no UI option for Instance Links yet. Use the setup script or the API directly.

### Required environment variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `ENABLE_DISAGGREGATION=true` | **Both nodes** | Without this, every `/v1/instance-links` endpoint 404s. |
| `EXO_STREAMING_PREFILL=1` | **Prefill node** (DGX Spark) | Enables layer-by-layer KV streaming during prefill. Without it, disaggregated prefill still works but falls back to computing the whole prefill before sending anything — no compute/network overlap. |
| `EXO_REMOTE_PREFILL_MIN_TOKENS` | **Decode node** (Mac Studio) | **Read this even if you skip everything else below.** Default `1000`. The decode node only calls out to the linked prefill node if the *uncached* portion of the prompt exceeds this many tokens — below it, the decode node just runs prefill itself and the prefill node is never contacted at all. Lower it for testing with short prompts, e.g. `EXO_REMOTE_PREFILL_MIN_TOKENS=50 uv run exo`. |

`ENABLE_DISAGGREGATION` and `EXO_STREAMING_PREFILL` are already set in `docker-compose.yml`
for the DGX container. On the Mac, export the ones relevant to whichever role it's
playing before `uv run exo` — `EXO_STREAMING_PREFILL` only matters on the prefill node,
`EXO_REMOTE_PREFILL_MIN_TOKENS` only matters on the decode node.

### Option A: automated setup script (recommended)

`scripts/setup_disaggregated_cluster.py` drives the whole flow — pins one instance to
each node (`/instance/previews?node_ids=...`, the only placement endpoint that can
target a specific node), waits for both to load, links them, and optionally sends a
test request. Safe to re-run: it reuses any existing instance/link instead of creating
duplicates.

```bash
# Run from either node — talks to the cluster over HTTP, no local exo import needed
# beyond httpx (already a project dependency)
uv run python scripts/setup_disaggregated_cluster.py \
  --model mlx-community/Qwen3.6-35B-A3B-bf16 \
  --prefill-host spark-ams01:52415 \
  --decode-host moysas-mac-studio:52415 \
  --test
```

This assumes the model is already downloaded on both nodes. It prints each step
(node ID resolution, instance creation, runner status transitions, linking) and fails
loudly with a specific error if `ENABLE_DISAGGREGATION` isn't set, if a placement has no
valid single-node option, or if a runner reports `RunnerFailed`.

### Option B: manual API walkthrough

Useful for understanding what the script does, or for debugging a specific step.

```bash
API=http://localhost:52415   # any node's API — cluster state is shared
MODEL="mlx-community/Qwen3.6-35B-A3B-bf16"

# 0. Confirm the flag actually took
curl -s $API/v1/feature-flags   # must show {"disaggregation": true}

# 1. Get node IDs directly from each node (do not guess — hardware identity
#    strings are unreliable on Linux today, see src/exo/utils/info_gatherer/system_info.py)
NODE_DGX=$(curl -s http://spark-ams01:52415/node_id | tr -d '"')
NODE_MAC=$(curl -s http://moysas-mac-studio:52415/node_id | tr -d '"')

# 2. Preview a placement PINNED to each node (the only placement endpoint that
#    can target a specific node — /place_instance cannot be pinned)
curl -s "$API/instance/previews?model_id=$MODEL&node_ids=$NODE_DGX" \
  | jq -c '.previews[] | select(.error==null) | .instance' | head -1 > /tmp/dgx.json
curl -s "$API/instance/previews?model_id=$MODEL&node_ids=$NODE_MAC" \
  | jq -c '.previews[] | select(.error==null) | .instance' | head -1 > /tmp/mac.json

# 3. Create both instances verbatim from the previews
curl -X POST $API/instance -H 'Content-Type: application/json' \
  -d "{\"instance\": $(cat /tmp/dgx.json)}"
curl -X POST $API/instance -H 'Content-Type: application/json' \
  -d "{\"instance\": $(cat /tmp/mac.json)}"

# 4. Wait for both, then find their instance IDs by matching node ID
curl -N "$API/instance/await?model_id=$MODEL"
curl -s $API/state | jq '.instances[] | {id: .instanceId, nodes: (.shardAssignments.nodeToRunner | keys)}'
# Set from the output above:
PREFILL_ID="<instance id whose nodes includes $NODE_DGX>"
DECODE_ID="<instance id whose nodes includes $NODE_MAC>"

# 5. Link them: DGX instance = prefill source, Mac instance = decode target
curl -X POST $API/v1/instance-links -H 'Content-Type: application/json' \
  -d "{\"prefill_instances\": [\"$PREFILL_ID\"], \"decode_instances\": [\"$DECODE_ID\"]}"

# 6. Run inference — routing to the decode instance and pulling prefill from
#    the linked DGX instance is fully automatic based on model_id
curl -N -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"stream\": true}"
```

Request-body fields are snake_case (`prefill_instances`, `model_id`); response fields
on `/state`, `/instance/previews`'s nested `.instance`, and `/v1/instance-links` GET
responses come back camelCase (`shardAssignments`, `nodeToRunner`, `prefillInstances`).
Always copy the `.instance` object from a preview verbatim into the create call rather
than hand-writing it.

### Why "nothing reaches the DGX" is the expected result for short prompts

The single most common confusion testing this: everything looks correctly linked (the
`/advanced` page shows the route, the setup script succeeds, chat works), but 100% of
processing visibly happens on the Mac and the DGX is never touched. **This is not a
routing bug** — the master correctly excludes prefill-only instances from ordinary
generation dispatch. It's the `EXO_REMOTE_PREFILL_MIN_TOKENS` gate above: short chat
messages (a sentence or two) never have more than a handful of uncached tokens, so
`use_remote` (`src/exo/worker/engines/mlx/generator/generate.py`,
`.../batch_generate.py`) evaluates `False` and the decode node just runs prefill itself
— the prefill node is never even contacted, let alone streamed to.

This also means `scripts/setup_disaggregated_cluster.py --test`'s own built-in test
prompt ("Say hello in exactly 5 words", ~18 tokens) **only validates that instance
creation, pinning, and linking work — it never exercises the actual remote-prefill data
path.** A successful `--test` run proves the plumbing is correct, not that DGX did any
work.

To actually trigger and observe the remote path:

1. Either paste a long document/article as the prompt (comfortably over 1000
   uncached tokens), or lower the threshold for testing:
   `EXO_REMOTE_PREFILL_MIN_TOKENS=50 uv run exo` on the Mac (decode node).
2. Send a **fresh, unique** long prompt each time — repeating a similar prompt lets the
   Mac's own local `KVPrefixCache` absorb more of it on each attempt, shrinking the
   uncached count and making it progressively *less* likely to cross the threshold.
3. Tail the **prefill node's** logs during that request. You should see multiple
   `KVChunk` sends interleaved with prefill progress, not one burst at the very end
   (that's `EXO_STREAMING_PREFILL` actually doing something).
4. If `_StreamingKVLayer`'s defensive catch fires for a layer, it logs at `debug` level:
   `"Streaming prefill: layer N hook failed, will fall back to non-streamed send"` —
   run with `-vv` to see these.
5. Also watch the **decode node's** logs for `"Remote prefill failed, falling back to
   local prefill"` — a silent exception in the remote path looks identical to "never
   crossed the threshold" unless you're watching for it.
6. Compare total request latency with `EXO_STREAMING_PREFILL` unset vs `=1` for the same
   long prompt — streaming should win once DGX compute time and KV transfer time are
   both significant.

---

## 5. Running a Model (Single Multi-Node Instance)

This is the simpler default path — one instance spanning both nodes, chosen via the
dashboard's Tensor/Pipeline toggle or the placement API. **It does not give DGX-prefill/
Mac-decode specialization** (see section 4 above for that). Pipeline is still
meaningfully better than Tensor here: Tensor requires an all-reduce sync at every layer
for both prefill and decode; Pipeline only crosses the network once per forward pass, at
the single boundary between each node's layer range.

### Via the dashboard (simplest)

1. Open `http://localhost:52415/` on either node
2. Select a model or search on HuggingFace
3. Click "Load" — exo's placement engine automatically splits the model across both nodes

### Via the API

**Preview placements** to see what the planner proposes:

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b" \
  | jq '.previews[] | select(.error == null) | {model_id, sharding, instance_meta, memory_delta_by_node}'
```

**Create the instance:**

```bash
curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d '{"instance": {"model_id": "llama-3.2-1b", "placement": {}}}'
```

**Wait for the model to be ready:**

```bash
curl -N "http://localhost:52415/instance/await?model_id=llama-3.2-1b"
```

**Send a chat completion request:**

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "What is pipeline parallelism?"}],
    "stream": true
  }'
```

**Ollama-compatible API** (for tools like OpenWebUI):

```bash
curl http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Verify the placement

```bash
curl http://localhost:52415/state | jq '.instances'
```

Look for `start_layer` / `end_layer` in the shard metadata to confirm the model is split across both nodes.

---

## 6. Testing

Two independent things to verify: the code changes type-check/lint/pass their unit
tests (works on either platform, no cluster needed), and the live 2-node cluster
actually streams disaggregated prefill correctly (needs both nodes up).

### Unit tests — on the Mac (native MLX)

`mlx` is only installable on macOS/Apple Silicon or Linux+CUDA — it cannot be tested in
a plain Linux CPU sandbox, so this must run on real hardware.

```bash
cd ~/exo   # the audiohacking/exo checkout
git fetch origin && git checkout feature/linux-cuda-support && git pull

# Dashboard must be built or the whole suite fails to import
# (find_dashboard() raises FileNotFoundError otherwise)
cd dashboard && npm install && npm run build && cd ..

# Sync deps WITH the mlx extra. A plain `uv sync` is exact and will
# UNINSTALL mlx/mlx-lm/mlx-vlm (they live in the optional `mlx` extra) —
# if that happens, this command restores them.
uv sync --extra mlx

# Full suite (excludes slow/network/multi-process tests by default)
uv run pytest

# Just the streaming-prefill-specific tests, faster:
uv run pytest \
  src/exo/worker/tests/unittests/test_runner/test_serve_prefill.py \
  src/exo/worker/tests/unittests/test_runner/test_streaming_prefill.py \
  src/exo/worker/engines/mlx/disaggregated/tests/ \
  -v

# Required pre-commit checks
uv run basedpyright
uv run ruff check
nix fmt
```

What each file covers:

| File | Covers |
|------|--------|
| `disaggregated/tests/test_streaming_prefill.py` | `StreamingPrefillLayers` — fires the per-layer callback correctly, restores original layers on exit (including on exception), skips gracefully for layers with no `cache` kwarg |
| `test_runner/test_streaming_prefill.py` | `_serve_prefill_streaming`'s wire orchestration — one `KVChunk` per layer as it streams, the final-flush fallback when `on_layer_ready` never fires, `ArraysCache`/SSM state sent exactly once, `EXO_STREAMING_PREFILL` env var parsing, and `_serve_prefill` routing to the streaming vs. bulk path |
| `test_runner/test_serve_prefill.py` | The pre-existing bulk (non-streaming) path — unchanged behavior, since `on_layer_ready` defaults to `None` |
| `disaggregated/tests/test_mlx_adapter.py` | `build_kv_chunk_for_entry`/`send_arrays_cache_entry` — the functions extracted from `send_mlx_kv_cache` during the refactor; confirms it's behavior-preserving |
| `disaggregated/tests/test_end_to_end.py` (marked `@pytest.mark.slow`) | Real `PrefillServer`/socket round-trip — run with `uv run pytest -m ""` to include it |

These tests mock `run_prefill_for_request`/`mlx_prefill` rather than loading a real
model, so they run in a couple seconds and don't need a downloaded model or a second
node — they verify the orchestration logic (chunking, threading, wire format), not
inference correctness.

### Unit tests — on the DGX Spark (CUDA/Linux)

Same commands, but run **inside the container** where `mlx-cuda13` is installed (the
host Linux Python environment doesn't have `mlx` at all):

```bash
docker compose build   # picks up any source changes
docker compose run --rm exo bash

# now inside the container:
cd /app
uv run pytest src/exo/worker/tests/unittests/test_runner/test_streaming_prefill.py \
  src/exo/worker/engines/mlx/disaggregated/tests/ -v
```

The tests themselves are backend-agnostic (they use plain `KVCache`/`ArraysCache`
objects, not real CUDA/Metal tensors), so the same test files validate the logic on
both platforms — there's no CUDA-specific or Metal-specific test variant needed.

### Live end-to-end test (both nodes up)

Unit tests validate the orchestration logic in isolation; they don't prove the two real
nodes actually stream KV cache to each other correctly. For that, run the cluster:

1. Confirm `ENABLE_DISAGGREGATION=true` on both nodes and `EXO_STREAMING_PREFILL=1` on
   the DGX container (see section 4), then start both nodes and confirm they've
   discovered each other (section 3).
2. Run `scripts/setup_disaggregated_cluster.py --test` (section 4, Option A) — the
   `--test` flag sends a real chat completion once linked and prints the streamed
   response.
3. Tail the DGX container's logs during that request and confirm `KVChunk` sends are
   interleaved with prefill progress rather than arriving in one burst — see "Verifying
   streaming is actually active" in section 4.
4. For a real performance comparison, run the same prompt twice — once with
   `EXO_STREAMING_PREFILL` unset (bulk path) and once with it set to `1` — using a
   long enough prompt that DGX compute time and KV transfer time are both significant.
   Short prompts won't show a meaningful difference.

---

## 7. Understanding What Happened (Single Multi-Node Instance)

### Placement decision

When you requested a model, the master's `place_instance()` function:

1. Found a 2-node cycle in the topology graph
2. Filtered by memory (total RAM ≥ model size)
3. Filtered by backend compatibility — `MlxRing` requires each node to support at least one of `[MlxMetal, MlxCuda, MlxCpu]`. The Mac has `[MlxCpu, MlxMetal]`, the DGX has `[MlxCpu, MlxCuda]`, so `MlxCpu` is the common backend
4. Allocated layers proportionally to each node's available RAM

### Distributed initialization

The master generates per-node host lists and sets environment variables:

| Variable | Purpose |
|----------|---------|
| `MLX_HOSTS_JSON` | Per-node MLX ring host configuration |
| `MLX_HOSTFILE` | Path to a temp file copy of the host list |
| `MLX_RANK` | 0 for one node, 1 for the other |

Each node calls `mx.distributed.init(backend="ring", strict=True)` to form the MLX ring group.

### CUDA routing

On the DGX Spark, the `TcpRelay` component handles CUDA-to-CUDA communication:

- Started eagerly during distributed init (see `src/exo/worker/engines/mlx/utils_mlx.py`)
- Listens on port `40000 + rank`
- Routes CUDA-to-CUDA send/recv through raw TCP sockets (bypassing the broken MLX ring send/recv for CUDA aarch64)
- Metal-to-CUDA operations use the standard MLX ring backend

### Layer allocation

For a 2-node pipeline, layers are split proportionally to available RAM via `allocate_layers_proportionally()` in `src/exo/master/placement_utils.py`. The DGX Spark (128 GB) may get more layers than the Mac Studio depending on available memory. The DGX Spark handles prefill (computing KV caches for the prompt), then streams them layer-by-layer to the Mac Studio for decode (token-by-token generation).

---

## 8. Troubleshooting

### Model fails to load

- Check memory: `curl http://localhost:52415/state | jq '.nodes[].memory'`
- Check placement preview: look for `"error": null` in the response
- The model must support the selected sharding strategy

### Slow performance

- **Network speed is the #1 factor.** A direct 10 GbE or switched 10 GbE link between the Mac and DGX is required. Wi-Fi or 1 GbE will bottleneck the KV cache transfer and eliminate the speedup.
- Verify both nodes are connected in the dashboard
- Check shard assignments in `/state` to confirm the model is split across both nodes

### CUDA TcpRelay fails

- Check logs for `CUDA TcpRelay server started on port 40000`
- Verify NVIDIA driver is loaded: `nvidia-smi` should work
- Check for port conflicts: `ss -tlnp | grep 4000`

### API returns 502 or connection refused

- The API listens on port `52415`. Verify it is running: `curl http://localhost:52415/node_id`
- If running from Docker, use the DGX Spark's IP address instead of `localhost`

---

## 9. Speculative Decoding (DFlash / MTP drafters)

Ported from upstream [PR #2079](https://github.com/exo-explore/exo/pull/2079)
(collocated drafting only — the asymmetric remote-drafter placement was not
ported). A model card may declare a **coupled drafter** (`coupled_drafter` in
the card TOML): a small model that consumes the target's hidden states each
draft step and proposes token blocks the target verifies in one forward pass.
Lossless — output is identical to normal decoding. Upstream benchmarked
Qwen3.6-35B-A3B-8bit + z-lab DFlash at **4.30× decode speedup** (92.6%
acceptance) on Apple Silicon.

### How it activates

1. The model card declares `coupled_drafter` (already set on the
   `mlx-community/Qwen3.6-35B-A3B-8bit` and `-bf16` cards — 8bit is the
   upstream-benchmarked pairing) or `drafter_model_ids` (standard external
   drafter sharing the target's tokenizer).
2. The drafter weights must already be on disk — automatic drafter download is
   NOT ported. Pre-download with:
   `uv run python scripts/download_model_to_cluster.py z-lab/Qwen3.6-35B-A3B-DFlash --host <node>`
   If absent, the runner logs a warning and falls back to plain decoding.
3. On instance load, the runner loads the drafter (coupled drafters via
   mlx-vlm ≥ 0.5.0), attaches target-side hooks, and selects the
   SequentialGenerator (the batch engine has no speculative hook).
4. Verify in logs: `Loaded coupled drafter ... kind='dflash'` and
   `using SequentialGenerator (coupled drafter loaded: ...)`.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `EXO_DRAFT_MODE` | `model` / `pipelined` / `ngram` / `none` — default `model` when a drafter loaded, else `none` |
| `EXO_DISABLE_DRAFTER` | `1` skips drafter loading entirely |
| `EXO_NUM_DRAFT_TOKENS` | Draft block size K (default 5) |
| `EXO_ADAPTIVE_DRAFT_TOKENS` | `1` adapts K per-round from rolling acceptance |
| `EXO_DRAFTER_MIN_OUTPUT_TOKENS` | Skip drafting for requests with max_tokens at or below this (default 16) |
| `EXO_DRAFTER_PREFERENCE` | `fastest` / `highest_acceptance` / `auto` for multi-entry `drafter_model_ids` |

Per-request overrides on `/v1/chat/completions`: `use_drafter`,
`num_draft_tokens`, `draft_mode`. Telemetry lands in the response's
`generation_stats` (accepted/proposed draft tokens, acceptance fraction,
drafter kind).

### Platform notes

- **Mac (Metal)**: the upstream-validated path.
- **DGX/CUDA**: unproven — the drafter graph runs on the same MLX ops the
  target uses, so it should execute under `mlx-cuda13`, but correctness must
  be verified (lossless ⇒ output must match plain decoding byte-for-byte)
  and speedup re-measured. mlx-vlm must also install against mlx-cuda13.
- Speculative decoding composes with disaggregated prefill (section 4): the
  decode node runs the drafter; remote prefill still supplies the KV cache.
  This combination is also unvalidated — test each feature separately first.

---

## 10. Reference

### Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 52413 | UDP | mDNS discovery |
| 52414 | TCP | Zenoh router |
| 52415 | TCP | API + dashboard |
| 40000+rank | TCP | TcpRelay (CUDA-to-CUDA data transfer) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `EXO_ZENOH_NAMESPACE` | Cluster namespace (default: exo version string) |
| `EXO_MODELS_DIRS` | Model download directories (colon-separated) |
| `EXO_OFFLINE` | Skip internet checks (`true`/`false`) |
| `ENABLE_DISAGGREGATION` | Enables `/v1/instance-links` (prefill/decode disaggregation) — required on both nodes for section 4 |
| `EXO_STREAMING_PREFILL` | Enables layer-by-layer KV streaming during prefill — set on the prefill node only |
| `EXO_REMOTE_PREFILL_MIN_TOKENS` | Uncached-token threshold before the decode node bothers calling the linked prefill node (default `1000`) — set on the decode node only |
| `MLX_HOSTS_JSON` | MLX ring host configuration (set automatically) |
| `MLX_RANK` | MLX distributed rank (set automatically) |
| `MLX_CUDA_RANKS` | Comma-separated CUDA ranks (set automatically) |

### File Paths (DGX Spark / Linux)

| Path | Purpose |
|------|---------|
| `~/.local/share/exo/models/` | Model weights |
| `~/.cache/exo/` | Cache and logs |
| `~/.config/exo/` | Configuration |
