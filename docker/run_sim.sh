#!/usr/bin/env bash
# Inside the vagen_src_sim container: virtual display -> AMBF -> env server (text observation).
#   docker run -d --name vagen_sim --network host --ipc host -v <repo>:/repo \
#     -e PORT=8001 -e TASK=approach vagen_src_sim bash docker/run_sim.sh
set -uo pipefail
PORT="${PORT:-8001}"; TASK="${TASK:-approach}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}" DISPLAY=:99
mkdir -p /repo/results
Xvfb :99 -screen 0 1280x1024x24 >/dev/null 2>&1 &
sleep 2
bash /repo/scripts/launch_sim.sh > /repo/results/ambf_docker.log 2>&1 &
source /repo/scripts/sim_env.sh
# The scene takes minutes to load; the env server needs the PSM objects to be there.
for _ in $(seq 120); do
  ros2 topic list 2>/dev/null | grep -q "/ambf/env/psm2/baselink/State" && break
  sleep 5
done
sleep 30
cd /repo
exec "$SIM_PYTHON" -m vagen_src.serve --task "$TASK" --port "$PORT" --camera_topic ""
