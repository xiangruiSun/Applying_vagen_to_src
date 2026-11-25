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

## Results on `approach` (2026-11-25, RTX 4080 16 GB)

All rows: the same 100 random-needle seeds (20000-20099, `expert_start: false`), SurgicAI's
success thresholds, 300 low-level step budget. Trajectory length and steps are means over
successful episodes. Raw per-episode data, the training curves and the validation rollouts
are in [`results/approach_20261125/`](results/approach_20261125).

| Method | Success | Trajectory length (mm) | Low-level steps | Model turns |
|---|---|---|---|---|
| SurgicAI `BC` | 100/100 | 87.1 +- 16.4 | 57.3 +- 10.2 | - |
| SurgicAI `TD3_HER_BC` | 100/100 | 104.0 +- 44.0 | 73.9 +- 32.2 | - |
| Qwen3-0.6B untrained, `compact` prompt | 49/100 | 37.8 +- 14.6 | 62.4 +- 16.0 | 1.92 |
| Qwen3-0.6B trained `compact` (step 30) | 100/100 | 35.1 +- 5.2 | 91.0 +- 23.8 | 4.48 |
| Qwen3-0.6B untrained, `rag` prompt | 100/100 | 35.5 +- 7.3 | 61.5 +- 14.1 | 2.06 |
| Qwen3-0.6B trained `rag` (step 60) | 100/100 | 35.9 +- 6.2 | 92.9 +- 25.1 | 4.76 |

- GRPO training takes the hint-free `compact` model from 49/100 to 100/100 on random needle
  poses, although it only trained from the fixed expert start.
- With the expert hint the untrained model is already at 100/100, so `rag` training cannot
  improve success on this task.
- Training makes episodes longer: mean turns per training episode rise from 2.3 to 5.9 over
  the `rag` run (`training/training_metrics.csv`), and the trained models need ~4.6 turns /
  ~92 steps against ~2 turns / ~62 steps untrained. Cause not confirmed; the per-turn
  `format_reward` (0.1) is a candidate.
- The in-training validation (`configs/val_*`) uses `expert_start: true`, which switches the
  needle noise off: its 16 episodes all start from the same scene. Use the
  `configs/eval_src_approach_text_{compact,rag}.yaml` configs above for generalisation.

What was run, and what differs from the commands below:

- `LR=5e-6`, not `2e-5`. At `2e-5` the model stopped closing `</answer>` after one update;
  every episode then scored 0, the gradient was 0, and the run could not recover.
- `compact` stopped at step 32 of 60 (another user's job filled the shared GPU; out of
  memory in the actor update). The evaluated checkpoint is `best_actor` (step 30). `rag` ran
  all 60 steps with `GPU_MEM=0.35`, which peaks near 9.5 GB on the 16 GB card.
- With the conda CUDA toolkit, SGLang's flashinfer build fails with `cannot find -lcuda`.
  Fix: `mkdir -p ~/vagen_src/cuda_link && ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1
  ~/vagen_src/cuda_link/libcuda.so` and `export LIBRARY_PATH=~/vagen_src/cuda_link` before
  `train_grpo.sh` / `serve_model.sh`.
- After a trainer is killed mid-episode the env server keeps its session for
  `--session_timeout` (600 s); the next run waits that long before its first episode.
- `scripts/eval_surgicai_baseline.py` now passes the env when loading a checkpoint
  (`TD3_HER_BC` needs it for its HER replay buffer).
- Evaluating a checkpoint: serve it with
  `MODEL_PATH=<run>/verl_checkpoints/.../huggingface MEM_FRACTION=0.3 MAX_MODEL_LEN=4096 bash scripts/serve_model.sh`,
  then `MODEL=<same path> bash scripts/eval_vagen.sh configs/eval_src_approach_text_compact.yaml envs.0.harness=no_concat`.
  `MODEL` must be the real model id or path: the evaluator loads its tokenizer from it.
- Checkpoints (15 GB) are not in the repository.

## Status and how to continue on a new machine (2026-11-25)

**Where the work stands**

- Zero-shot Qwen2.5-VL-3B on `approach`: 0/10. The task needs training.
- RL training (GRPO, `scripts/train_grpo.sh`) runs end to end with Qwen3-0.6B on text
  observations. Run 1 (long WM prompt, RTX 3090) reached a rising score but no success in
  its first 8 steps; its final validation was never read back.
- New on 2026-11-25, verified in the simulator but **not yet trained**: the compact prompt,
  expert retrieval and expert start described [below](#compact-prompt-expert-retrieval-expert-start-2026-11-25).
- **Next step:** train `VARIANT=compact` (baseline, no expert hint), then `VARIANT=rag`, and
  compare validation success. Then try a larger text model (Qwen3-1.7B) if memory allows.
- **GPU memory:** Qwen3-0.6B used ~14.5 GB on a 24 GB RTX 3090 (`GPU_MEM=0.5`). It does not
  fit an 8 GB card: the SGLang rollout engine needs `GPU_MEM` >= ~0.33 of 8 GB and the
  actor update then runs out of memory. Qwen2.5-VL-3B does not fit 24 GB. Plan for >= 16 GB.
- **Open question:** SurgicAI's approach expert trajectories do not fit this scene (replayed
  verbatim they end 29 mm / 97 deg from the needle), so only the arm start is matched.

**One machine with a GPU, from scratch** (Ubuntu 24.04, ROS 2 Jazzy, NVIDIA driver with CUDA 13)

```bash
# 0. system packages
sudo apt install ros-jazzy-desktop python3-venv python3-pykdl ros-jazzy-python-orocos-kdl-vendor \
     ros-jazzy-cv-bridge ros-jazzy-image-transport-plugins libasound2-dev xorg-dev libusb-1.0-0-dev \
     libyaml-cpp-dev libboost-all-dev

# 1. AMBF 3.0 (the commit the SurgicAI ROS 2 fork was tested with)
mkdir -p ~/ros_ambf_ws/src && cd ~/ros_ambf_ws/src
git clone -b ambf-3.0 https://github.com/WPI-AIM/ambf.git && git -C ambf checkout da2fd987
cd ~/ros_ambf_ws && source /opt/ros/jazzy/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 2. this repo with SurgicAI (old SRC scene + expert trajectories) and VAGEN (+ verl)
mkdir -p ~/vagen_src && cd ~/vagen_src
git clone --recursive https://github.com/xiangruiSun/Applying_vagen_to_src.git && cd Applying_vagen_to_src

# 3. simulator-side Python (system site-packages for rclpy / PyKDL)
python3 -m venv --system-site-packages ~/vagen_src/venv_sim
~/vagen_src/venv_sim/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/vagen_src/venv_sim/bin/pip install -r requirements-sim.txt
~/vagen_src/venv_sim/bin/python -m pytest tests -q           # 15 offline tests, no simulator needed

# 4. training-side Python (VAGEN main + verl + SGLang; needs nvcc and gcc at run time)
python3 -m venv ~/vagen_src/venv_vagen && source ~/vagen_src/venv_vagen/bin/activate
(cd third_party/VAGEN && bash scripts/install.sh) && pip install httpx fastapi pytest

# 5. simulator: AMBF window first, wait ~3 min for the scene, then the env server
bash scripts/launch_sim.sh                                              # terminal 1
bash scripts/serve_sim.sh --task approach --port 8000 --camera_topic "" # terminal 2 (text-only)

# 6. check without any model: both should report 16/16
python scripts/eval_scripted.py --policy offset --seeds 10000 10015
python scripts/eval_scripted.py --policy rag    --seeds 10000 10015

# 7. train (terminal 3, venv_vagen active). Baseline first, then with the expert hint.
VAGEN_DIR=$PWD/third_party/VAGEN VARIANT=compact LR=2e-5 TOTAL_STEPS=60 bash scripts/train_grpo.sh
VAGEN_DIR=$PWD/third_party/VAGEN VARIANT=rag     LR=2e-5 TOTAL_STEPS=60 bash scripts/train_grpo.sh
# validation success every 10 steps:
grep -o "step:[0-9]* .*" exps/vagen_src/*/run.log | tr ' ' '\n' | grep -E "^step:|val.*success"
```

Things that cost time before:

- `ambf_simulator -g 0` (headless) segfaults with this scene; keep the window, or use Xvfb as
  `docker/run_sim.sh` does (`docker/Dockerfile.sim` builds the whole simulator side in one image).
- Start the env server only after the scene has loaded (~3 min); earlier it crashes or hangs.
- One simulator runs one episode at a time, so a training step of `BATCH * ROLLOUT_N` (4 x 4)
  episodes takes minutes. Do not let the machine suspend during a run.
- If the simulator and the GPU are on different machines, open
  `ssh -N -R 8000:localhost:8000 <gpu-host>` from the simulator machine and run step 7 there.
- `GPU_MEM` is SGLang's share of the whole card, training copy included; do not set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (SGLang refuses to start).
- LoRA with verl's SGLang rollout is broken in this stack; the script fine-tunes all weights.

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

## Compact prompt, expert retrieval, expert start (2026-11-25)

Four changes aimed at making a small Qwen model trainable on `approach`:

1. **Angles are roll/pitch/yaw, never quaternions.** SurgicAI's state is already
   `(x, y, z, roll, pitch, yaw, jaw)`; every pose and offset the model sees is six numbers
   in mm / deg (asserted in `tests/test_fake_sim.py`).
2. **Compact prompt** (`prompt_format: compact`). The observation is the offset to the
   target (target minus gripper) -- the number an action consumes -- and the reply is the
   action alone: `<answer>move(..)|rotate(..)|gripper(close)</answer>`. With the
   `no_concat` harness each turn is its own conversation, so the only part of a reply that
   reaches the next turn is the executed `<answer>`, echoed as `Last action:`. Per turn
   (Qwen tokenizer): system 445 -> 316, observation 286 -> 114, reply ~154 -> 42 tokens.
3. **Expert retrieval** (`rag_hint: true`, `vagen_src/expert_rag.py`). SurgicAI's expert
   trajectories (`RL/Expert_traj/<Task>/all_episodes_merged.pkl`, 50 episodes per task)
   are indexed by offset-to-goal. Each turn the nearest expert state is looked up and the
   motion that reaches the expert's state `rag_lookahead` (15) low-level steps later is
   shown as `Expert hint: ...`. `scripts/eval_scripted.py` runs the hint, or plain offset
   cancelling, as a policy without any model:

   | policy (approach, old SRC scene) | success | turns | low-level steps |
   |---|---|---|---|
   | expert hint, random needle (seeds 10000-10015) | 16/16 | 2.9 | 71 |
   | expert hint, expert start | 8/8 | 3.0 | 69 |
   | cancel the offset, random needle | 16/16 | 2.0 | 61 |

4. **Expert start** (`expert_start: true`). Every episode starts with the arm at the
   expert demonstrations' start pose (exact: SurgicAI's `init_psm2`) and the needle at its
   nominal pose with SurgicAI's needle noise off. The needle is *not* where it was in the
   demonstrations: their recorded target is 27 mm / 112 deg from the target of a needle
   lying on the phantom in the old SRC scene, and the needle pose that would reproduce it
   is 10 mm above the phantom and tilted ~80 deg, so the demonstrations were recorded in
   another scene setup. The gap is returned in the reset info (`expert_start_error`);
   `expert_start_check_target: true` turns it into an error. Retrieval is unaffected
   because it works on goal-relative offsets.

Training: `VARIANT=rag bash scripts/train_grpo.sh` (or `VARIANT=compact` for no hint),
configs `configs/{train,val}_src_approach_text_{rag,compact}.yaml`.

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
