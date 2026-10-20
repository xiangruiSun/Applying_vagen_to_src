"""
SRCSim: one SurgicAI subtask env bound to one running AMBF simulator.

The VLM speaks in mid-level primitives (move / rotate / gripper). SRCSim turns
each primitive into a sequence of SurgicAI low-level actions in [-1, 1]^7 and
feeds them through SurgicAI's own `env.step`, so:
  - dynamics, workspace handling and success criteria are SurgicAI's;
  - "time cost" (low-level steps) and "trajectory length" (mm) are counted
    exactly like SurgicAI's Model_evaluation.py, making results comparable.

All methods are blocking and must run on the single worker thread returned by
`executor` (AMBF / ROS handles are not thread-safe). The constructor must run
on the main thread because the AMBF client calls rospy.init_node().
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .parse import Primitive
from .tasks import ARM_NAMES, load_surgicai_env_class, normalize_task

# Same per-step scaling SurgicAI uses for evaluation (RL/Model_evaluation.py).
STEP_SIZE = np.array(
    [1.0e-3, 1.0e-3, 1.0e-3, np.deg2rad(3), np.deg2rad(3), np.deg2rad(3), 0.05], dtype=np.float32
)
JAW_OPEN = 0.8   # SurgicAI's initial jaw angle
JAW_CLOSED = 0.0
# Tasks whose SurgicAI success criterion also requires jaw <= 0.1.
JAW_CLOSE_TASKS = ("approach", "regrasp")


class SRCSim:
    def __init__(
        self,
        task: str = "approach",
        trans_error: float = 0.5,
        angle_error: float = 30.0,
        max_low_level_steps: int = 300,
        camera_topic: Optional[str] = "/ambf/env/cameras/cameraL/ImageData",
        image_size: Tuple[int, int] = (448, 336),
        flip_image: bool = False,
    ):
        """
        Args:
            task: SurgicAI subtask (approach | place | insert | regrasp | pullout).
            trans_error: success translation threshold in cm (SurgicAI's --trans_error).
            angle_error: success angle threshold in degrees (SurgicAI's --angle_error).
            max_low_level_steps: episode budget, SurgicAI evaluates with 300.
            camera_topic: ROS image topic for vision observations, None for text-only.
        """
        self.task = normalize_task(task)
        self.trans_error = float(trans_error)
        self.angle_error = float(angle_error)
        self.max_low_level_steps = int(max_low_level_steps)

        env_cls = load_surgicai_env_class(self.task)
        threshold = np.array([self.trans_error, np.deg2rad(self.angle_error)], dtype=np.float32)
        self.env = env_cls(
            seed=None,
            render_mode=None,
            reward_type="sparse",
            threshold=threshold,
            max_episode_step=self.max_low_level_steps,
            step_size=STEP_SIZE,
        )

        self.camera = None
        if camera_topic:
            from .camera import RosCamera

            self.camera = RosCamera(camera_topic, resize=image_size, flip_vertical=flip_image)

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"src-{self.task}")
        self._reset_counters()

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _reset_counters(self) -> None:
        self.steps = 0
        self.traj_length_mm = 0.0
        self.success = False
        self.success_step: Optional[int] = None

    @property
    def psm_idx(self) -> int:
        return int(self.env.psm_idx)

    @property
    def active_arm(self) -> str:
        return ARM_NAMES[self.psm_idx]

    @property
    def needs_jaw_close(self) -> bool:
        return self.task in JAW_CLOSE_TASKS

    @property
    def steps_left(self) -> int:
        return max(0, self.max_low_level_steps - self.steps)

    @property
    def done(self) -> bool:
        return self.success or self.steps_left == 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> Dict[str, Any]:
        # SurgicAI's Approach/Place envs draw needle noise from the global
        # numpy RNG and ignore reset(seed=...), so seed it here.
        np.random.seed(seed % (2 ** 32))
        random.seed(seed)
        self.env.reset(seed=seed)
        self._reset_counters()
        return self.state()

    def low_level_step(self, action: np.ndarray) -> bool:
        """One SurgicAI env step. Returns True if the subtask succeeded."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        _, _, terminated, _, info = self.env.step(action)
        self.steps += 1
        # Identical to SurgicAI Model_evaluation.run_evaluation.
        self.traj_length_mm += float(np.linalg.norm(action[0:3] * STEP_SIZE[0:3] * 1000))
        if terminated or info.get("is_success", False):
            self.success = True
            self.success_step = self.steps
        return self.success

    def execute(self, prim: Primitive, max_move_mm: float, max_rotate_deg: float) -> int:
        """Expand one primitive into low-level steps. Returns steps executed."""
        delta = np.zeros(7, dtype=np.float32)
        if prim.name == "move":
            mm = np.clip(np.array(prim.values, dtype=np.float32), -max_move_mm, max_move_mm)
            delta[0:3] = mm / 1000.0
        elif prim.name == "rotate":
            deg = np.clip(np.array(prim.values, dtype=np.float32), -max_rotate_deg, max_rotate_deg)
            delta[3:6] = np.deg2rad(deg)
        elif prim.name == "gripper":
            target = JAW_OPEN if prim.jaw_open else JAW_CLOSED
            delta[6] = target - self._jaw()
        else:
            raise ValueError(f"Unknown primitive {prim.name}")

        ratio = np.abs(delta) / STEP_SIZE
        n = int(math.ceil(float(ratio.max()) - 1e-6))
        if n <= 0:
            return 0
        unit = delta / (n * STEP_SIZE)
        executed = 0
        for _ in range(n):
            if self.done:
                break
            self.low_level_step(unit)
            executed += 1
        return executed

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _jaw(self) -> float:
        return float(self.env.psm_goal_list[self.psm_idx - 1][-1])

    def state(self) -> Dict[str, Any]:
        """Poses in the active arm's base frame, converted to mm / degrees."""
        current = np.asarray(self.env.psm_goal_list[self.psm_idx - 1], dtype=np.float64)
        goal = np.asarray(self.env.goal_obs, dtype=np.float64)
        trans_mm = float(np.linalg.norm(goal[0:3] - current[0:3]) * 1000.0)
        angle_deg = float(np.degrees(np.linalg.norm(goal[3:6] - current[3:6])))
        return {
            "gripper_pose": _to_mm_deg(current),
            "target_pose": _to_mm_deg(goal),
            "jaw": float(current[6]),
            "trans_error_mm": trans_mm,
            "angle_error_deg": angle_deg,
            "success": self.success,
            "low_level_steps": self.steps,
            "steps_left": self.steps_left,
            "traj_length_mm": self.traj_length_mm,
        }

    def image(self, timeout: float = 5.0):
        if self.camera is None:
            return None
        # Give AMBF a moment to render the last command before grabbing.
        return self.camera.grab(timeout=timeout, newer_than=time.time() - 0.05)


def _to_mm_deg(pose: np.ndarray) -> list:
    return [float(v) for v in pose[0:3] * 1000.0] + [float(v) for v in np.degrees(pose[3:6])]
