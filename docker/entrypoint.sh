#!/usr/bin/env bash
set -euo pipefail

# Add any NVIDIA CUDA libraries bundled inside site-packages to LD_LIBRARY_PATH.
# This is required so that the MLX CUDA backend can find cuBLAS, cuDNN, etc.
# at runtime when they are installed as Python package data.
site_packages="$('/app/.venv/bin/python' - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

for library_dir in "$site_packages"/nvidia/*/lib "$site_packages"/nvidia/cu13/lib; do
  if [ -d "$library_dir" ]; then
    export LD_LIBRARY_PATH="$library_dir:${LD_LIBRARY_PATH:-}"
  fi
done

exec "$@"
