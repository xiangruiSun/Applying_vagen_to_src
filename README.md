# Applying VAGEN to the Surgical Robotics Challenge

MLRL final project: train a vision-language model (VLM) agent with
[VAGEN-Lite](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite) (multi-turn RL for VLM agents) on the
suturing subtasks of the [Surgical Robotics Challenge](https://github.com/surgical-robotics-ai/surgical_robotics_challenge)
(SRC), and compare it with [SurgicAI](https://github.com/surgical-robotics-ai/SurgicAI)'s RL / imitation policies
as the baseline.

This repo only contains the **mid-level glue code**. The upstream projects are git submodules:

| Submodule | Upstream | Pinned at | Why |
|---|---|---|---|
| `third_party/VAGEN` | mll-lab-nu/VAGEN | `main` @ `41df5e7` (VAGEN-Lite on verl 0.9) | evaluator, RL trainer, remote-env client/server |
| `third_party/SurgicAI` | xiangruiSun/SurgicAI | `jin-hierarchical-repro` @ `1dcc427` | subtask gym envs + baseline checkpoints; this fork also runs on **ROS 2 Jazzy / AMBF 3.0** through its `RL/ros1_compat` rospy shim |

SurgicAI's envs need the *old* SRC (`kinematics.psmFK`, `utils.task3_init`). The fork carries it as its own
submodule `surgical_robotics_challenge_old` (JinnWu686 fork @ `d2faceac`), so clone recursively:

```bash
git clone --recursive https://github.com/xiangruiSun/Applying_vagen_to_src.git
# or, after a plain clone:
git submodule update --init --recursive
```

## How it fits together

```
 GPU machine (RTX 3090, Python 3.12)             Simulator machine (Ubuntu 24.04, ROS 2 Jazzy, AMBF 3.0)
┌──────────────────────────────┐              ┌────────────────────────────────────────────────────────┐
│ SGLang server: the VLM       │ ◀── chat ─── │ VAGEN evaluator  (python -m vagen.evaluation)          │
│   scripts/serve_model.sh     │  (SSH tunnel)│   env name: RemoteEnv ── HTTP ──▶ vagen_src.serve      │
└──────────────────────────────┘              │   SRCEnv  : text + camera image obs, parses VLM answer │
                                              │   SRCSim  : primitives -> SurgicAI low-level steps     │
                                              │   SurgicAI <Task>_env  ──ROS──▶  AMBF + old SRC scene  │
                                              └────────────────────────────────────────────────────────┘
```

* The simulator side needs ROS + SB3 2.2.1 / gymnasium 0.29.1, the VAGEN side needs Torch 2.11 / gymnasium 1.x,
  so the environment runs as a separate HTTP service using VAGEN's own remote-env framework. VAGEN is used
  unmodified: configs just use the built-in `RemoteEnv`.
* The env wraps SurgicAI's subtask envs **without modification** (`approach`, `place`, `insert`, `regrasp`,
  `pullout`), so the VLM and the baseline share the reset distribution, dynamics and success criterion.
* The same code runs on a ROS 1 Noetic host: `paths.setup_sim_paths()` delegates to the fork's
  `hierarchical_paths.setup_sim_imports()`, which only enables the rospy shim when real `rospy` is absent.

### Action space (mid-level)

A VLM turn contains up to 3 primitives, e.g. `<answer>move(8, 5, -10)|rotate(5, 0, -3)</answer>`. Responses use
VAGEN's shared protocol (`<perception>…</perception><reasoning>…</reasoning><prediction>…</prediction><answer>…</answer>`
for `prompt_format: wm`, `<think>…</think><answer>…</answer>` for `free_think`):

| Primitive | Meaning |
|---|---|
| `move(dx, dy, dz)` | translate active gripper, mm, base frame (clipped to ±20 mm) |
| `rotate(dr, dp, dy)` | roll/pitch/yaw increment, degrees (clipped to ±30°) |
| `gripper(open)` / `gripper(close)` | jaw to 0.8 / 0.0 |

Each primitive is expanded into SurgicAI's low-level actions (1 mm, 3°, 0.05 jaw per step) and executed through
SurgicAI's `env.step`. Episodes end on success or after 300 low-level steps (SurgicAI's evaluation budget).

### Observation

Camera image (`/ambf/env/cameras/cameraL/ImageData/compressed`) + text with the active arm's pose and, if
`show_target: true`, the target pose. SurgicAI's state-based policies observe the goal pose, so
`show_target: true` is the like-for-like setting; `show_target: false` makes the VLM localise the target visually.

### Reward

`format_reward` (0.1) for a well-formed answer + progress shaping (0.05 / mm and 0.02 / deg of error reduced)
+ `success_reward` (10) on success.

### Metrics (identical to SurgicAI `RL/Model_evaluation.py`)

Success rate, trajectory length in mm (successful episodes) and time cost in low-level steps (successful
episodes). VAGEN additionally reports VLM turns.

## Repository layout

```
vagen_src/                 glue package
  paths.py                 puts third_party/* on sys.path, enables the ROS 2 shim when needed
  tasks.py                 SurgicAI task registry + instructions
  parse.py, prompt.py      VLM response parsing / prompts (VAGEN response protocol)
  camera.py                ROS image grabber (compressed topic by default)
  sim.py                   SRCSim: SurgicAI env + primitive execution + metrics
  src_env.py               SRCEnv: VAGEN GymImageEnv
  handler.py, serve.py     VAGEN remote env server
configs/                   VAGEN train / val / eval specs (approach subtask; *_text = text-only observation)
scripts/
  sim_env.sh               ROS 2 environment for the simulator side (sourced by the others)
  launch_sim.sh            AMBF with the old SRC scene (SurgicAI's launch flags + light frame)
  serve_sim.sh             start the env server
  serve_model.sh           SGLang VLM server (GPU machine)
  smoke_test_env.py        scripted end-to-end test, no VLM needed
  eval_vagen.sh            VAGEN evaluation against a served model
  eval_surgicai_baseline.py  SurgicAI pretrained policies on the same seeds
  compare_results.py       results table
  train_grpo.sh            VAGEN RL training (trajectory_grpo, one 24 GB GPU; see notes)
tests/test_fake_sim.py     offline test of the glue with a fake SurgicAI env
```

## Setup

### 1. Simulator machine (tested: Ubuntu 24.04, ROS 2 Jazzy, colcon-built AMBF 3.0 in `~/ros_ambf_ws`)

```bash
# AMBF: see third_party/SurgicAI/README.md section 1 (branch ambf-3.0, colcon build)
python3 -m venv --system-site-packages ~/vagen_src/venv_sim        # system site-packages for rclpy / PyKDL
~/vagen_src/venv_sim/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/vagen_src/venv_sim/bin/pip install -r requirements-sim.txt

bash scripts/launch_sim.sh                                          # terminal 1 (AMBF window)
source scripts/sim_env.sh
$SIM_PYTHON scripts/smoke_test_env.py --mode local --task approach  # optional check, no VLM
bash scripts/serve_sim.sh --task approach --port 8000               # terminal 2
```

`scripts/sim_env.sh` sets `ROS_DOMAIN_ID=42` (keeps this scene apart from other AMBF / dVRK nodes),
`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`, and `SIM_PYTHON` (override any of them from the environment).
One simulator runs one episode at a time. For parallel rollouts launch more simulators (each with its own
`ROS_DOMAIN_ID`) and servers on other ports, then list all of them in `base_urls`.

### 2. VAGEN (evaluator on the simulator machine, model server on the GPU machine)

Install VAGEN `main` in a Python 3.12 environment on both (CUDA 13 driver required):
```bash
python3 -m venv ~/vagen_src/venv_vagen && source ~/vagen_src/venv_vagen/bin/activate   # or conda, python=3.12
cd third_party/VAGEN && bash scripts/install.sh && pip install httpx fastapi pytest
```
The GPU machine needs `nvcc` and a C compiler at *run* time (flashinfer / triton compile kernels when SGLang
starts) and to build `causal-conv1d`. Without root:
`conda install -c nvidia -c conda-forge cuda-toolkit=13.0 gcc_linux-64 gxx_linux-64`, then `CUDA_HOME=$CONDA_PREFIX`.

```bash
# GPU machine
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/serve_model.sh
# simulator machine
ssh -N -L 30000:localhost:30000 <user>@<gpu-host> &
python scripts/smoke_test_env.py --mode remote --base_url http://localhost:8000
```

## Experiments

```bash
# Baseline: SurgicAI pretrained policies (simulator machine, env server stopped)
$SIM_PYTHON scripts/eval_surgicai_baseline.py --task approach --algorithm TD3_HER_BC --seeds_from configs/eval_src_approach.yaml
$SIM_PYTHON scripts/eval_surgicai_baseline.py --task approach --algorithm BC         --seeds_from configs/eval_src_approach.yaml

# Zero-shot VLM (model served on the GPU machine, tunnel open, env server running)
MODEL=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/eval_vagen.sh configs/eval_src_approach.yaml envs.0.n_envs=10

# RL training (GPU machine; env server reachable at the config's base_urls, e.g. via
#   ssh -N -R 8000:localhost:8000 <user>@<gpu-host>   from the simulator machine)
VAGEN_DIR=/path/to/VAGEN LR=2e-5 TOTAL_STEPS=60 bash scripts/train_grpo.sh trainer.test_freq=10 trainer.save_freq=20

# Comparison table
python3 scripts/compare_results.py \
  --vagen results/vagen_eval/src_approach/Qwen_Qwen2.5-VL-3B-Instruct \
  --baseline results/baseline/approach_TD3_HER_BC.json results/baseline/approach_BC.json \
  --out results/compare_approach.csv
```

Other subtasks: copy the `configs/*_approach.yaml` files, change `task:` and start the server with
`--task <name>`.

## Notes / known gaps

* Success thresholds default to SurgicAI's env defaults (`--trans_error 0.5` cm, `--angle_error 30` deg); use the
  same values for the server and the baseline script.
* SurgicAI's reported numbers (e.g. Approach TD3+HER+BC: 96% success) are in
  `third_party/SurgicAI/RL/Evaluation_model/*/*/*_results.txt`; re-running the baseline here gives numbers on the
  exact seeds the VLM sees.
* SurgicAI hard-codes some absolute paths (`/home/jin/...`) in its high-level / multi-task scripts; the subtask
  envs used here do not depend on them.
* **Status (2026-11-13).** Verified: offline glue tests; scripted-oracle smoke test on the live AMBF scene
  (in-process and through `RemoteEnv`); zero-shot VLM evaluation on approach; one full RL training step
  (rollout on the live scene + update) with `scripts/train_grpo.sh`. **Not yet run:** the SurgicAI baseline
  script, the other four subtasks, `compare_results.py` against VAGEN main's output layout. A 60-step training
  run was started on 2026-11-13; no trained-model result yet.
* **Zero-shot Qwen2.5-VL-3B on approach: 0/10** (seeds 20000-20009, 12 turns each; position error never
  shrank, 31-42 mm at start vs 35-140 mm at the end). The model answers a template (`move(10, 0, 0)`, early
  `gripper(close)`, `stop`, empty answers) instead of using the pose numbers. A task-specific system prompt
  with the explicit offset rule did not fix single-step answers either. The same model scores 24/128 (18.75%)
  on VAGEN's stock Sokoban evaluation.
* **Training on one 24 GB GPU.** `scripts/train_grpo.sh` runs `trajectory_grpo` + `no_concat` with
  `Qwen/Qwen3-0.6B` on the text observation (`configs/*_src_approach_text.yaml`), full fine-tune. Qwen2.5-VL-3B
  does not fit with this stack (rollout engine + training copy exceed 24 GB at every
  `gpu_memory_utilization` tried), and LoRA with verl's SGLang rollout failed in three places; details are in
  the script's comments. Measured on the smoke step: ~27 s per 8-turn episode, one simulator, so a step costs
  about `train_batch_size * rollout.n * 27 s`.
* AMBF needs ~3 minutes to load the scene; start `serve_sim.sh` after that or it times out waiting for the
  first camera frame.
* ROS 2: raw 640x480 camera frames (0.9 MB) are sometimes never delivered to a client process, so the camera
  reads the `/compressed` topic. Without ADF 18 (`light_frame.yaml`) the rendered images are almost black.
