# Applying VAGEN to the Surgical Robotics Challenge

MLRL final project: train a vision-language model (VLM) agent with
[VAGEN-Lite](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite) (multi-turn RL for VLM agents) on the
suturing subtasks of the [Surgical Robotics Challenge](https://github.com/surgical-robotics-ai/surgical_robotics_challenge)
(SRC), and compare it with [SurgicAI](https://github.com/surgical-robotics-ai/SurgicAI)'s RL / imitation policies
as the baseline.

This repo only contains the **mid-level glue code**. The three upstream projects are pulled in as git submodules,
the same way SurgicAI references SRC:

| Submodule | Upstream | Pinned at | Why |
|---|---|---|---|
| `third_party/VAGEN` | mll-lab-nu/VAGEN | `vagen-lite` branch | VLM multi-turn RL trainer, evaluator, remote-env client/server |
| `third_party/SurgicAI` | surgical-robotics-ai/SurgicAI | `b7ce546` | Subtask gym envs + pretrained baseline checkpoints |
| `third_party/surgical_robotics_challenge` | surgical-robotics-ai/surgical_robotics_challenge | `03befbf` (v2.0.0) | SRC, the version SurgicAI pins as `_new` |
| `third_party/surgical_robotics_challenge_old` | JinnWu686/surgical_robotics_challenge | `d2faceac` | SRC fork SurgicAI pins as `_old`; **its RL envs import `kinematics.psmFK` and `utils.task3_init`, which only exist here**, so this is the one used at runtime |

```bash
git clone --recursive https://github.com/xiangruiSun/Applying_vagen_to_src.git
# or, after a plain clone:
git submodule update --init --recursive
```

## How it fits together

```
 GPU machine (Python 3.12)                      Simulator machine (Ubuntu 20.04, ROS Noetic, Python 3.8)
┌──────────────────────────────┐   HTTP     ┌────────────────────────────────────────────────────────┐
│ VAGEN trainer / evaluator    │──────────▶ │ vagen_src.serve   (VAGEN GymService + SRCHandler)      │
│   env name: RemoteEnv        │ ◀──────────│   SRCEnv  : text + camera image obs, parses VLM answer │
│   (GymImageEnvClient)        │ obs/reward │   SRCSim  : primitives -> SurgicAI low-level steps     │
└──────────────────────────────┘            │   SurgicAI <Task>_env  ──ROS──▶  AMBF + SRC            │
                                            └────────────────────────────────────────────────────────┘
```

* SRC/AMBF needs ROS 1 + Python 3.8, while VAGEN needs Python ≥ 3.10, so the environment runs as a separate
  HTTP service using VAGEN's own remote-env framework (`vagen/envs_remote`). VAGEN is used unmodified: training
  and evaluation configs just use the built-in `RemoteEnv`.
* The env wraps SurgicAI's subtask envs **without modification** (`approach`, `place`, `insert`, `regrasp`,
  `pullout`), so the VLM and the baseline share the reset distribution, dynamics and success criterion.

### Action space (mid-level)

A VLM turn contains up to 3 primitives, e.g. `<answer>move(8, 5, -10)|rotate(5, 0, -3)</answer>`:

| Primitive | Meaning |
|---|---|
| `move(dx, dy, dz)` | translate active gripper, mm, base frame (clipped to ±20 mm) |
| `rotate(dr, dp, dy)` | roll/pitch/yaw increment, degrees (clipped to ±30°) |
| `gripper(open)` / `gripper(close)` | jaw to 0.8 / 0.0 |

Each primitive is expanded into SurgicAI's low-level actions (1 mm, 3°, 0.05 jaw per step) and executed through
SurgicAI's `env.step`. Episodes end on success or after 300 low-level steps (SurgicAI's evaluation budget).

### Observation

Camera image (`/ambf/env/cameras/cameraL/ImageData`) + text with the active arm's pose and, if
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
  paths.py                 puts third_party/* on sys.path (overridable via env vars)
  tasks.py                 SurgicAI task registry + instructions
  parse.py, prompt.py      VLM response parsing / prompts (VAGEN primitive_skill style)
  camera.py                ROS 1 image grabber
  sim.py                   SRCSim: SurgicAI env + primitive execution + metrics
  src_env.py               SRCEnv: VAGEN GymImageEnv
  handler.py, serve.py     VAGEN remote env server
configs/                   VAGEN train / val / eval specs (approach subtask)
scripts/
  launch_sim.sh            roscore + AMBF with SRC (SurgicAI's launch flags)
  serve_sim.sh             start the env server
  smoke_test_env.py        scripted end-to-end test, no VLM needed
  train_ppo_qwen25vl3b.sh  VAGEN PPO training
  eval_vagen.sh            VAGEN evaluation (API model or trained checkpoint)
  eval_surgicai_baseline.py  SurgicAI pretrained policies on the same seeds
  compare_results.py       results table
tests/test_fake_sim.py     offline test of the glue with a fake SurgicAI env
```

## Setup

### 1. Simulator machine (Ubuntu 20.04, ROS Noetic, Python 3.8)

1. Install [AMBF 2.0](https://github.com/WPI-AIM/ambf) and ROS Noetic as described in the SRC README
   (`third_party/surgical_robotics_challenge_old`), then install the SRC python package from the **old** fork:
   ```bash
   cd third_party/surgical_robotics_challenge_old/scripts && pip3 install -e .
   ```
2. Python deps for SurgicAI + the env server:
   ```bash
   pip3 install -r requirements-sim.txt
   ```
3. Run:
   ```bash
   bash scripts/launch_sim.sh                                # terminal 1
   python3 scripts/smoke_test_env.py --mode local --task approach   # optional check
   bash scripts/serve_sim.sh --task approach --port 8000     # terminal 2
   ```
   One simulator runs one episode at a time. For parallel rollouts launch more simulators (each with its own
   `ROS_MASTER_URI`) and servers on other ports, then list all of them in `base_urls`.

### 2. GPU machine (training / VLM evaluation)

Follow VAGEN's install inside the submodule:
```bash
conda create -n vagen python=3.12 -y && conda activate vagen
cd third_party/VAGEN
cd verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh && pip install --no-deps -e . && cd ..
pip install -e . && pip install "trl==0.26.2"
cd ../..
python scripts/smoke_test_env.py --mode remote --base_url http://<sim-host>:8000
```
Set `base_urls` in `configs/*.yaml` to `http://<sim-host>:8000` (or tunnel with `ssh -L 8000:localhost:8000`).

## Experiments

```bash
# Baseline: SurgicAI pretrained policies (simulator machine, server stopped)
python3 scripts/eval_surgicai_baseline.py --task approach --algorithm TD3_HER_BC --seeds_from configs/eval_src_approach.yaml
python3 scripts/eval_surgicai_baseline.py --task approach --algorithm BC         --seeds_from configs/eval_src_approach.yaml

# Zero-shot VLM before RL (any OpenAI-compatible backend)
bash scripts/eval_vagen.sh configs/eval_src_approach.yaml run.backend=openai backends.openai.model=gpt-4.1-mini

# VAGEN training (GPU machine)
N_GPUS=4 TASK=approach bash scripts/train_ppo_qwen25vl3b.sh

# Trained checkpoint, served with vLLM
bash scripts/eval_vagen.sh configs/eval_src_approach.yaml run.backend=vllm backends.vllm.model=<served-name>

# Comparison table
python3 scripts/compare_results.py \
  --vagen results/vagen_eval/src_approach=VAGEN-Qwen2.5VL-3B \
  --baseline results/baseline/approach_TD3_HER_BC.json results/baseline/approach_BC.json \
  --out results/compare_approach.csv
```

Other subtasks: copy the three `configs/*_approach.yaml` files, change `task:` and start the server with
`--task <name>`.

## Notes / known gaps

* Success thresholds default to SurgicAI's env defaults (`--trans_error 0.5` cm, `--angle_error 30` deg); use the
  same values for the server and the baseline script.
* SurgicAI's reported numbers (e.g. Approach TD3+HER+BC: 96% success) are in
  `third_party/SurgicAI/RL/Evaluation_model/*/*/*_results.txt`; re-running the baseline here gives numbers on the
  exact seeds the VLM sees.
* SurgicAI hard-codes some absolute paths (`/home/jin/...`) in its high-level / multi-task scripts; the subtask
  envs used here do not depend on them.
