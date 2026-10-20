#!/bin/bash
# VAGEN PPO training of Qwen2.5-VL-3B on SRC (adapted from VAGEN's
# examples/train/primitive_skill/train_ppo_qwen25vl3b.sh).
#
# 1. On the simulator machine(s): bash scripts/launch_sim.sh
#                                 bash scripts/serve_sim.sh --task approach --port 8000
# 2. Point base_urls in configs/train_src_approach.yaml / val_src_approach.yaml at the server(s).
# 3. On the GPU machine:          bash scripts/train_ppo_qwen25vl3b.sh
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAGEN_DIR=${REPO_ROOT}/third_party/VAGEN
TASK=${TASK:-approach}
N_GPUS=${N_GPUS:-4}

PROJECT_NAME="vagen_src"
EXPERIMENT_NAME="src_${TASK}_ppo_qwen25vl3b"
EXPERIMENT_DIR=${REPO_ROOT}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${REPO_ROOT}/configs/train_src_${TASK}.yaml
DATASET_VAL=${REPO_ROOT}/configs/val_src_${TASK}.yaml
agent_loop_config_path=${VAGEN_DIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=${REF_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
mkdir -p ${EXPERIMENT_DIR}

cd ${VAGEN_DIR}
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${VAGEN_DIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=32 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=16 \
    data.max_prompt_length=3000 \
    data.max_response_length=6000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.total_training_steps=${TOTAL_STEPS:-100} 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log
