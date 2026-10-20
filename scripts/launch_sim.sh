#!/usr/bin/env bash
# Start roscore + the AMBF simulator with SRC, the same way SurgicAI does.
# Override paths with env vars if your AMBF / SRC live elsewhere:
#   AMBF_DIR=~/ambf SRC_DIR=~/ambf/surgical_robotics_challenge bash scripts/launch_sim.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMBF_DIR="${AMBF_DIR:-$HOME/ambf}"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/third_party/surgical_robotics_challenge_old}"
# SurgicAI's launch: phantom(0) ghosts(1) PSM1(3) PSM2(4) camera(13) ... see SRC launch.yaml
LAUNCH_IDS="${LAUNCH_IDS:-0,1,3,4,13,14}"

if ! rostopic list >/dev/null 2>&1; then
  roscore & sleep 3
fi

"$AMBF_DIR/bin/lin-x86_64/ambf_simulator" \
  --launch_file "$SRC_DIR/launch.yaml" \
  -l "$LAUNCH_IDS" -p 200 -t 1 \
  --override_max_comm_freq 120 --override_min_comm_freq 120 "$@"
