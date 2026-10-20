"""
SurgicAI subtask registry.

SurgicAI decomposes suturing into five low-level subtasks, each a gymnasium
env in ``third_party/SurgicAI/RL/<Task>_env.py`` with class ``SRC_<task>``
(same naming rule as SurgicAI's ``Model_evaluation.py``). We reuse those envs
unchanged so the VLM agent and the SurgicAI baseline face the same reset
distribution, dynamics and success criteria.
"""

from __future__ import annotations

from typing import Dict

TASK_INSTRUCTIONS: Dict[str, str] = {
    "approach": (
        "PSM2 (the right arm) starts with an open gripper. Move PSM2's gripper to a grasping "
        "point on the suturing needle lying on the tissue phantom and close the jaw to grasp it."
    ),
    "place": (
        "PSM2 is holding the needle. Move the needle so that its tip reaches the entry point "
        "on the tissue phantom with the correct orientation for insertion."
    ),
    "insert": (
        "PSM2 is holding the needle with its tip at the entry point. Drive the needle through "
        "the tissue so the tip comes out of the exit point."
    ),
    "regrasp": (
        "The needle tip is sticking out of the exit point. Move PSM1 (the left arm) to grasp "
        "the needle near its tip and close the jaw."
    ),
    "pullout": (
        "PSM1 is holding the needle tip. Pull the needle out of the tissue and bring it to the "
        "handover pose above the exit point."
    ),
}

ARM_NAMES = {1: "PSM1 (left arm)", 2: "PSM2 (right arm)"}


def normalize_task(name: str) -> str:
    key = name.strip().lower()
    if key not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown SRC task {name!r}. Available: {sorted(TASK_INSTRUCTIONS)}")
    return key


def load_surgicai_env_class(task: str):
    """Import SurgicAI's env class for `task` (requires paths.setup_sim_paths())."""
    import importlib

    task = normalize_task(task)
    module = importlib.import_module(f"{task.capitalize()}_env")
    return getattr(module, f"SRC_{task}")
