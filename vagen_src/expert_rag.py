"""
Retrieval over SurgicAI's expert trajectories ("RAG" for robot motion).

SurgicAI ships expert demonstrations per subtask in
``third_party/SurgicAI/RL/Expert_traj/<Task>/all_episodes_merged.pkl``: a flat list of
transitions ``{obs, next_obs, action, reward, done, info}`` whose ``achieved_goal`` /
``desired_goal`` are ``(x, y, z, roll, pitch, yaw, jaw)`` in cm / rad.

Every expert state is stored as its *offset to the goal* (goal minus gripper, mm / deg),
the same quantity the agent sees. A query finds the expert state nearest to the current
offset and returns where the expert was ``lookahead`` low-level steps later. The
suggested motion is the one that puts the gripper on that expert state:

    delta = current_offset - expert_offset[i + lookahead]

so the suggestion both follows the expert's path and pulls a gripper that is off the
path back onto it; since every expert episode ends at the goal, repeated suggestions
converge. Offsets are goal-relative, which is what lets demonstrations recorded for one
needle pose be reused for another.

Only numpy is needed, so the index runs on the simulator side inside ``SRCEnv``.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .parse import Primitive

#: A position error of 1 mm weighs as much as this many degrees in the nearest-state search.
DEG_PER_MM = 3.0
#: Expert jaw angle below which the expert has started closing (0.8 open, 0.0 closed).
JAW_CLOSING = 0.75
#: Motions smaller than this are left out of the suggestion.
MIN_MOVE_MM = 0.05
MIN_ROTATE_DEG = 0.05


def default_expert_path(task: str) -> str:
    from .paths import get_path

    return os.path.join(get_path("SURGICAI_RL_DIR"), "Expert_traj", task.capitalize(), "all_episodes_merged.pkl")


def _to_mm_deg(goal7: np.ndarray) -> np.ndarray:
    out = np.asarray(goal7, dtype=np.float64).copy()
    out[..., 0:3] *= 10.0                      # cm -> mm
    out[..., 3:6] = np.degrees(out[..., 3:6])  # rad -> deg
    return out


@dataclass
class Retrieval:
    primitives: List[Primitive]
    match_distance_mm: float     # weighted distance to the matched expert state
    expert_index: int            # flat index of the matched state
    steps_to_end: int            # expert low-level steps left after the returned state

    def __str__(self) -> str:
        return "|".join(str(p) for p in self.primitives)


class ExpertIndex:
    """Nearest-neighbour index over expert states of one subtask."""

    def __init__(self, transitions: Sequence[Dict[str, Any]], lookahead: int = 15):
        if not transitions:
            raise ValueError("no expert transitions")
        self.lookahead = int(lookahead)
        offsets, jaws, ends = [], [], []
        start = 0
        for i, tr in enumerate(transitions):
            last = bool(np.asarray(tr["done"]).ravel()[0]) or i == len(transitions) - 1
            if not last:
                # Merged files only mark `done` on success; also split where the next
                # transition does not continue this one.
                nxt = np.asarray(transitions[i + 1]["obs"]["achieved_goal"], dtype=np.float64)
                cur = np.asarray(tr["next_obs"]["achieved_goal"], dtype=np.float64)
                last = bool(np.abs(nxt - cur).max() > 1e-4)
            if last:
                episode = transitions[start:i + 1]
                # States s_0 .. s_T: the observation of every step, then the final next_obs.
                states = [(t["obs"]["achieved_goal"], t["obs"]["desired_goal"]) for t in episode]
                states.append((episode[-1]["next_obs"]["achieved_goal"], episode[-1]["next_obs"]["desired_goal"]))
                first = len(offsets)
                for achieved, desired in states:
                    a, d = _to_mm_deg(achieved), _to_mm_deg(desired)
                    offsets.append((d - a)[0:6])
                    jaws.append(float(a[6]))
                ends.extend([first + len(states) - 1] * len(states))
                start = i + 1
        first = transitions[0]["obs"]
        #: Where the demonstrations start, in SRCSim.state() units (mm / deg).
        self.first_state = {
            "gripper_pose": [float(v) for v in _to_mm_deg(first["achieved_goal"])[0:6]],
            "target_pose": [float(v) for v in _to_mm_deg(first["desired_goal"])[0:6]],
        }
        self.offsets = np.asarray(offsets, dtype=np.float64)   # (N, 6) mm / deg
        self.jaws = np.asarray(jaws, dtype=np.float64)          # (N,)
        self.episode_end = np.asarray(ends, dtype=np.int64)     # (N,) flat index of the episode's last state
        self.n_episodes = int(len(np.unique(self.episode_end)))
        self._scale = np.array([1.0, 1.0, 1.0, 1.0 / DEG_PER_MM, 1.0 / DEG_PER_MM, 1.0 / DEG_PER_MM])
        self._scaled = self.offsets * self._scale

    @classmethod
    def load(cls, path: str, lookahead: int = 15) -> "ExpertIndex":
        with open(path, "rb") as f:
            return cls(pickle.load(f), lookahead=lookahead)

    def __len__(self) -> int:
        return len(self.offsets)

    def query(
        self,
        offset_mm_deg: Sequence[float],
        jaw: float,
        max_move_mm: float,
        max_rotate_deg: float,
        lookahead: Optional[int] = None,
    ) -> Retrieval:
        """Suggest primitives for a gripper whose offset to the goal is `offset_mm_deg`."""
        k = self.lookahead if lookahead is None else int(lookahead)
        cur = np.asarray(offset_mm_deg, dtype=np.float64)[0:6]
        dist = np.linalg.norm(self._scaled - cur * self._scale, axis=1)
        i = int(np.argmin(dist))
        end = int(self.episode_end[i])
        j = min(i + max(k, 1), end)
        delta = cur - self.offsets[j]
        move = np.clip(delta[0:3], -max_move_mm, max_move_mm)
        rot = np.clip(delta[3:6], -max_rotate_deg, max_rotate_deg)

        prims: List[Primitive] = []
        if np.abs(move).max() >= MIN_MOVE_MM:
            prims.append(Primitive("move", tuple(round(float(v), 1) for v in move)))
        if np.abs(rot).max() >= MIN_ROTATE_DEG:
            prims.append(Primitive("rotate", tuple(round(float(v), 1) for v in rot)))
        # Close only once the motion of this turn is not cut short by the per-turn limits:
        # the expert closes the jaw when it has arrived, not on the way.
        reaches = bool(np.allclose(move, delta[0:3]) and np.allclose(rot, delta[3:6]))
        if self.jaws[j] < JAW_CLOSING and reaches and jaw > 0.1:
            prims.append(Primitive("gripper", jaw_open=False))
        if not prims:
            prims.append(Primitive("gripper", jaw_open=False))
        return Retrieval(prims, float(dist[i]), i, end - j)
