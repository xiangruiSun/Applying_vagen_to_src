#!/usr/bin/env bash
# Evaluate a VLM on SRC through VAGEN's evaluator. The model is any server that
# speaks the OpenAI/OpenRouter chat format (SGLang / vLLM on the GPU machine, or
# a hosted API). The env server must be running (scripts/serve_sim.sh).
#   MODEL_URL=http://localhost:30000/v1 MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
#     bash scripts/eval_vagen.sh configs/eval_src_approach.yaml [omegaconf overrides]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/eval_src_approach.yaml}"
shift 2>/dev/null || true
MODEL_URL="${MODEL_URL:-http://localhost:30000/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/third_party/VAGEN:$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p results
python -m vagen.evaluation --config "$CONFIG" \
  backends.openrouter.base_url="$MODEL_URL" backends.openrouter.model="$MODEL" \
  experiment.dump_dir="$REPO_ROOT/results/vagen_eval/$(basename "$CONFIG" .yaml | sed s/^eval_//)/${MODEL//\//_}" \
  "$@" 2>&1 | tee "results/eval_$(date +%Y%m%d_%H%M%S).log"
