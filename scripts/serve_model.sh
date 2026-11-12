#!/usr/bin/env bash
# Serve a VLM with SGLang on the GPU machine (run inside the VAGEN environment).
# The evaluator on the simulator machine reaches it through an SSH tunnel:
#   ssh -N -L 30000:localhost:30000 <user>@<gpu-host>
#
#   MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/serve_model.sh
#   MODEL_PATH=/path/to/verl_checkpoints/global_step_N/actor/huggingface bash scripts/serve_model.sh
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# flashinfer compiles its kernels at startup and needs nvcc + a C compiler. With a
# conda-provided toolkit (conda install -c nvidia -c conda-forge cuda-toolkit=13.0
# gcc_linux-64 gxx_linux-64) that is the environment prefix.
if [ -z "${CUDA_HOME:-}" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
  export CUDA_HOME="$CONDA_PREFIX"
fi
# Bound to localhost on purpose: reach it through the SSH tunnel, not the LAN.
exec python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$MAX_MODEL_LEN" \
  --random-seed 42 --enable-deterministic-inference \
  --trust-remote-code --log-level warning "$@"
