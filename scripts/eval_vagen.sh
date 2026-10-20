#!/usr/bin/env bash
# Evaluate a VLM (API model, or a VAGEN checkpoint served by vLLM/SGLang) on SRC.
# The env server must be running (scripts/serve_sim.sh).
#   bash scripts/eval_vagen.sh configs/eval_src_approach.yaml \
#        run.backend=vllm backends.vllm.model=<served-model-name>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/eval_src_approach.yaml}"
shift 2>/dev/null || true
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/third_party/VAGEN:$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p results
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@" 2>&1 | tee "results/eval_$(date +%Y%m%d_%H%M%S).log"
