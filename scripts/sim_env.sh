#!/usr/bin/env bash
# Sourced by the simulator-side scripts. ROS 2 host: source ROS + the colcon AMBF
# workspace and pick a private ROS_DOMAIN_ID so this scene does not mix with other
# AMBF / dVRK nodes on the machine. ROS 1 host: nothing to do (run roscore yourself).
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-jazzy}"
AMBF_WS="${AMBF_WS:-$HOME/ros_ambf_ws}"
if [ -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ] && [ -f "$AMBF_WS/install/setup.bash" ]; then
  set +u
  source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
  source "$AMBF_WS/install/setup.bash"
  set -u 2>/dev/null || true
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
  # FastDDS shared-memory transport leaves stale segments when a client dies and
  # then silently drops large topics (camera images) for later clients. Use UDP.
  export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
  export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
fi
SIM_PYTHON="${SIM_PYTHON:-$HOME/vagen_src/venv_sim/bin/python}"
[ -x "$SIM_PYTHON" ] || SIM_PYTHON=python3
