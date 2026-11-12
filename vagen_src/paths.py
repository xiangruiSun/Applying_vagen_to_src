"""
Make the third_party submodules importable without pip-installing them.

SurgicAI is not a Python package: its env modules (e.g. ``Place_env``) import
each other by bare name (``from subtask_env import SRC_subtask``) and need the
*old* SRC python package (``kinematics.psmFK``, ``utils.task3_init``), which the
SurgicAI fork carries as its own submodule ``surgical_robotics_challenge_old``.

The fork's ``RL/hierarchical_paths.setup_sim_imports()`` puts that old SRC
first on ``sys.path`` and, on a ROS 2 host (no real ``rospy``), enables its
``ros1_compat`` rospy shim + ambf_client patches. We delegate to it, so the same
code runs on ROS 1 Noetic and on ROS 2 Jazzy.

Locations can be overridden with environment variables:
    VAGEN_DIR=~/VAGEN  SURGICAI_RL_DIR=~/SurgicAI/RL
"""

from __future__ import annotations

import os
import sys
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")

_DEFAULTS: Dict[str, str] = {
    "SURGICAI_RL_DIR": os.path.join(THIRD_PARTY, "SurgicAI", "RL"),
    "VAGEN_DIR": os.path.join(THIRD_PARTY, "VAGEN"),
}


def get_path(name: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(name, _DEFAULTS[name])))


def _prepend(name: str) -> str:
    path = get_path(name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{name}={path} does not exist. Run `git submodule update --init --recursive` "
            f"or export {name} to point at your checkout."
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def setup_vagen_path() -> None:
    """Path needed on the training / evaluation side (VAGEN only)."""
    _prepend("VAGEN_DIR")


def setup_sim_paths() -> bool:
    """Paths needed on the simulator side (old SRC + SurgicAI + VAGEN remote server).

    Returns True when the ROS 2 rospy shim is in use.
    """
    _prepend("VAGEN_DIR")
    _prepend("SURGICAI_RL_DIR")
    import hierarchical_paths  # from the SurgicAI fork

    _, shim = hierarchical_paths.setup_sim_imports()
    return bool(shim)
