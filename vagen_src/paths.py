"""
Make the third_party submodules importable without pip-installing them.

SurgicAI is not a Python package: its env modules (e.g. ``Place_env``) import
each other by bare name (``from subtask_env import SRC_subtask``) and import
``surgical_robotics_challenge`` from the SRC ``scripts/`` folder. We mirror
that by prepending the same folders to ``sys.path``.

Each location can be overridden with an environment variable, e.g. if SRC is
already installed in your AMBF workspace:
    SRC_SCRIPTS_DIR=~/ambf/surgical_robotics_challenge/scripts
"""

from __future__ import annotations

import os
import sys
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")

# SurgicAI's RL envs import `kinematics.psmFK` and `utils.task3_init`, which
# only exist in the SRC fork SurgicAI pins as `surgical_robotics_challenge_old`.
_DEFAULTS: Dict[str, str] = {
    "SRC_SCRIPTS_DIR": os.path.join(THIRD_PARTY, "surgical_robotics_challenge_old", "scripts"),
    "SURGICAI_RL_DIR": os.path.join(THIRD_PARTY, "SurgicAI", "RL"),
    "VAGEN_DIR": os.path.join(THIRD_PARTY, "VAGEN"),
}


def get_path(name: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(name, _DEFAULTS[name])))


def setup_sim_paths() -> None:
    """Paths needed on the simulator side (SRC + SurgicAI + VAGEN remote server)."""
    for name in ("VAGEN_DIR", "SURGICAI_RL_DIR", "SRC_SCRIPTS_DIR"):
        path = get_path(name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{name}={path} does not exist. Run `git submodule update --init --recursive` "
                f"or export {name} to point at your checkout."
            )
        if path not in sys.path:
            sys.path.insert(0, path)


def setup_vagen_path() -> None:
    """Path needed on the training / evaluation side (VAGEN only)."""
    path = get_path("VAGEN_DIR")
    if path not in sys.path:
        sys.path.insert(0, path)
