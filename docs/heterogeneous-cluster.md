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
| **Network** | Both machines on the same LAN (same subnet). Ethernet recommended. |

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
- The `exo:cuda13` Docker image (built from this repo — see [Setup](#4-setting-up-the-dgx-spark-node))

### Namespace

Both nodes must share the same `--namespace` (default is the exo version string `0.3.70`). Nodes with different namespaces will not discover each other.

---

## 1. Setting Up the Mac Studio Node

```bash
cd ~/exo

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

### Build the Docker image

On the DGX Spark, clone the repo and build the CUDA container:

```bash
git clone https://github.com/exo-explore/exo
cd exo

# Build the CUDA 13 image
docker compose build
```

The multi-stage Dockerfile builds:
1. **Rust builder** — compiles the `exo_rs` PyO3 wheel with maturin
2. **Dashboard builder** — compiles the Svelte frontend
3. **Runtime** — installs Python 3.13, syncs deps with `--extra mlx-cuda13`, installs the prebuilt wheel

### Run the container

```bash
docker compose up
```

The `docker-compose.yml` config:

| Setting | Purpose |
|---------|---------|
| `network_mode: host` | Shares the host network so zenoh discovery and TcpRelay ports work |
| `deploy.resources.reservations.devices` | Exposes the GPU via the NVIDIA Container Toolkit |
| Volume mounts | Maps `~/.local/share/exo`, `~/.cache/exo`, `~/.config/exo`, `~/.cache/huggingface` |

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
| Nodes don't discover each other | Verify both are on the same subnet; check firewall allows UDP 52413 and TCP 52414 |
| Namespace mismatch | Check startup logs for `EXO_ZENOH_NAMESPACE` — both must match |
| DGX Spark started first | It may have elected itself master. This is fine — the Mac will connect as a worker. Use `--force-master` on the Mac if you want it to be master. |
| Docker not finding GPU | Verify `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi` works |

---

## 4. Running a Model

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

## 5. Understanding What Happened

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

- Started eagerly during distributed init
- Listens on port `40000 + rank`
- Routes CUDA-to-CUDA send/recv through raw TCP sockets (bypassing the broken MLX ring send/recv for CUDA)
- Metal-to-CUDA operations use the standard MLX ring backend

---

## 6. Troubleshooting

### Model fails to load

- Check memory: `curl http://localhost:52415/state | jq '.nodes[].memory'`
- Check placement preview: look for `"error": null` in the response
- The model must support the selected sharding strategy

### Slow performance

- Ethernet is strongly recommended — Wi-Fi adds KV cache transfer latency
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

## 7. Reference

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
| `MLX_HOSTS_JSON` | MLX ring host configuration (set automatically) |
| `MLX_RANK` | MLX distributed rank (set automatically) |
| `MLX_CUDA_RANKS` | Comma-separated CUDA ranks (set automatically) |

### File Paths (DGX Spark / Linux)

| Path | Purpose |
|------|---------|
| `~/.local/share/exo/models/` | Model weights |
| `~/.cache/exo/` | Cache and logs |
| `~/.config/exo/` | Configuration |
