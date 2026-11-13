#!/bin/bash
# VAGEN (main, verl 0.9) RL training on an SRC subtask, sized for ONE 24 GB GPU:
# trajectory_grpo (no critic model) + the no_concat harness (every turn is its
# own short prompt; the SRC observation already carries the full state).
# Adapted from VAGEN's examples/train/sokoban/train_grpo_qwen3_06b_lora_topo_filter_text.sh
# and examples/train/frozenlake/train_ppo_no_concat_lora_qwen25vl3b.sh.
#
# Default: Qwen/Qwen3-0.6B on the text observation (configs/*_src_<task>_text.yaml).
# Qwen2.5-VL-3B does NOT fit a single 24 GB card with this stack: the rollout engine
# (>= ~8.7 GB) plus the training copy (> 12 GB during weight sync) exceed it at every
# gpu_memory_utilization tried (0.4, 0.52, 0.65, 0.8). For a vision model on a bigger
# machine: MODEL=Qwen/Qwen2.5-VL-3B-Instruct OBS=vision GPU_MEM=0.6 ...
#
# Run on the GPU machine, inside the VAGEN environment, with the env server reachable
# at the base_urls of the config (e.g. `ssh -N -R 8000:localhost:8000 <gpu-host>` from
# the simulator machine):
#   VAGEN_DIR=~/vagen_src/VAGEN bash scripts/train_grpo.sh [hydra overrides]
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="${VAGEN_DIR:-$REPO_ROOT/third_party/VAGEN}"
V="$(cd "$V" && pwd)"
TASK=${TASK:-approach}
PROJECT_NAME=${PROJECT_NAME:-vagen_src}
OBS=${OBS:-text}                       # text | vision
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
GPU_MEM=${GPU_MEM:-0.5}
SUFFIX=$([ "$OBS" = text ] && echo _text || echo "")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-src_${TASK}_${OBS}_grpo_$(basename "$MODEL")}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$REPO_ROOT/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
mkdir -p "$EXPERIMENT_DIR"

if [ -z "${VERL:-}" ]; then
    for d in "$V/verl" "$V/../verl"; do
        if [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ]; then VERL=$(cd "$d" && pwd); break; fi
    done
fi
[ -n "${VERL:-}" ] || { echo "verl not found under $V (git submodule update --init -- verl)" >&2; exit 1; }
export PYTHONPATH=$VERL:$V${PYTHONPATH:+:$PYTHONPATH}
# SGLang's flashinfer compiles kernels at startup (see scripts/serve_model.sh).
if [ -z "${CUDA_HOME:-}" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
fi
# Inside verl's SGLang actor even /bin/sh needs libcudart.so.13 (a preloaded library
# pulls it in); without it Triton's `/sbin/ldconfig -p` probe exits 127 and the engine
# dies while compiling its CUDA graphs. Make the conda toolkit's copy findable and tell
# Triton where the driver library is so it does not need ldconfig at all.
if [ -n "${CUDA_HOME:-}" ] && ls "$CUDA_HOME"/lib/libcudart.so.* >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -z "${TRITON_LIBCUDA_PATH:-}" ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
fi
# The training copy is bf16 (verl's default fp32 doubles its footprint).
# gpu_memory_utilization is SGLang's mem-fraction-static, a share of the WHOLE card that
# also counts the training copy already on it (~9 GB), hence well above the ~0.4 a 3B engine alone would need.
# LoRA is off by default (LORA_RANK=0, full fine-tune -- cheap for a 0.6B model). With
# verl's SGLang rollout on this stack the adapter path failed three ways in a row:
# target_modules=all-linear is iterated character by character by SGLang 0.5.13,
# rollout.layered_summon needs sharded FSDP (>1 GPU), and the adapter weight sync dies
# in wrap_lora_params (asdict on a non-dataclass peft config).
LORA_RANK=${LORA_RANK:-0}
LR=${LR:-5e-6}
LORA=()
if [ "$LORA_RANK" -gt 0 ]; then
    LORA=(actor_rollout_ref.model.lora_rank="$LORA_RANK" actor_rollout_ref.model.lora_alpha="$LORA_RANK"
          "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]")
fi
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/training_defaults.flags" | sed "s|\$V|$V|g")

# rollout.agent.num_workers must divide train_batch_size * rollout.n (verl's default is 8).
# One simulator = one episode at a time, so a step costs about
# train_batch_size * rollout.n episodes of wall-clock; keep the product small.
PYTHONUNBUFFERED=1 python3 -m vagen.training.main \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/training/dataset.py" \
    "${BASE[@]}" \
    data.train_files="$REPO_ROOT/configs/train_src_${TASK}${SUFFIX}.yaml" \
    data.val_files="$REPO_ROOT/configs/val_src_${TASK}${SUFFIX}.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    algorithm.adv_estimator=trajectory_grpo \
    trainer.harness=no_concat \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    data.max_prompt_length=2000 \
    data.max_response_length=800 \
    actor_rollout_ref.actor.optim.lr="$LR" \
    "${LORA[@]}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.max_num_batched_tokens=6000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_training_steps=${TOTAL_STEPS:-100} \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$EXPERIMENT_DIR/verl_checkpoints" \
    trainer.rollout_data_dir="$EXPERIMENT_DIR/rollout_data" \
    trainer.validation_data_dir="$EXPERIMENT_DIR/validation" \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
    "$@" \
    2>&1 | tee "$EXPERIMENT_DIR/run.log"
