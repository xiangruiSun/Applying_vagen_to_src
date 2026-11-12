#!/usr/bin/env bash
# Start AMBF with the *old* SRC scene SurgicAI's envs were written for (needle
# without thread), with the same launch flags SurgicAI uses.
#   bash scripts/launch_sim.sh              # GUI
#   bash scripts/launch_sim.sh -g 0         # extra args go to ambf_simulator
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/sim_env.sh"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/third_party/SurgicAI/surgical_robotics_challenge_old}"
# 0,1 = psm1 + actuators, 3,4 = psm2 + actuators, 13 = phantom_without_thread_multiView, 14 = ghosts
# (SurgicAI's list) + 18 = light_frame: the world's main light is parented to it; without
# it the camera images are almost black, which only matters to image-based policies.
LAUNCH_IDS="${LAUNCH_IDS:-0,1,3,4,13,14,18}"
cd "$SRC_DIR"
exec ambf_simulator --launch_file launch.yaml -l "$LAUNCH_IDS" -p 200 -t 1 \
  --override_max_comm_freq 120 --override_min_comm_freq 120 "$@"
