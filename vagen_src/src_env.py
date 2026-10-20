"""
SRCEnv: VAGEN GymImageEnv for SurgicAI's SRC suturing subtasks.

One VAGEN turn = one VLM response = up to `max_actions_per_step` mid-level
primitives, each expanded into SurgicAI low-level steps by SRCSim. An episode
ends on SurgicAI success or when the low-level step budget (300, as in
SurgicAI's evaluation) runs out; VAGEN's `max_turns` caps the turns.

This module is imported on the simulator machine (Python 3.8, ROS Noetic), so
it avoids Python 3.9+ only APIs such as asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv

from .parse import VALID_FORMATS, parse_primitives, parse_response
from .prompt import get_format_instruction, observation_text, system_prompt
from .sim import SRCSim
from .tasks import TASK_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)


@dataclass
class SRCEnvConfig:
    task: str = "approach"
    render_mode: str = "vision"         # vision | text
    prompt_format: str = "wm"           # wm | free_think
    max_actions_per_step: int = 3
    action_sep: str = "|"
    max_move_mm: float = 20.0
    max_rotate_deg: float = 30.0
    # Show the target pose as text. SurgicAI's state-based policies observe the
    # goal, so keep True for a like-for-like comparison; set False to force the
    # VLM to localise the target from the image.
    show_target: bool = True
    # Rewards
    format_reward: float = 0.1
    progress_reward_per_mm: float = 0.05
    progress_reward_per_deg: float = 0.02
    success_reward: float = 10.0
    image_placeholder: str = "<image>"


class SRCEnv(GymImageEnv):
    def __init__(
        self,
        env_config: Dict[str, Any],
        sim: SRCSim,
        on_close: Optional[Callable[[], None]] = None,
    ):
        super().__init__(env_config)
        valid = {f.name for f in fields(SRCEnvConfig)}
        self.cfg = SRCEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        if self.cfg.prompt_format not in VALID_FORMATS:
            raise ValueError(f"Unknown prompt_format {self.cfg.prompt_format}. Valid: {VALID_FORMATS}")
        if self.cfg.task.lower() != sim.task:
            raise ValueError(
                f"Client asked for task {self.cfg.task!r} but this server runs {sim.task!r}. "
                f"Start a server with --task {self.cfg.task}."
            )
        if self.cfg.render_mode == "vision" and sim.camera is None:
            raise ValueError("render_mode=vision requires the server to be started with a camera topic.")
        self.sim = sim
        self._on_close = on_close
        self._closed = False
        self._t0 = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, init: bool, valid_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        st = self.sim.state()
        vision = self.cfg.render_mode == "vision"
        ph = self.cfg.image_placeholder
        obs_str = observation_text(
            init=init,
            image_placeholder=ph if vision else None,
            instruction=TASK_INSTRUCTIONS[self.sim.task],
            active_arm=self.sim.active_arm,
            gripper_pose=st["gripper_pose"],
            jaw=st["jaw"],
            target_pose=st["target_pose"] if self.cfg.show_target else None,
            steps_left=st["steps_left"],
            valid_actions=valid_actions,
            format_prompt=get_format_instruction(
                self.cfg.prompt_format, self.cfg.max_actions_per_step, self.cfg.action_sep
            ),
        )
        obs: Dict[str, Any] = {"obs_str": obs_str}
        if vision:
            obs["multi_modal_input"] = {ph: [self.sim.image()]}
        return obs

    # ------------------------------------------------------------------
    # Sync implementations (run on the simulator thread)
    # ------------------------------------------------------------------

    def _sync_reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        st = self.sim.reset(seed)
        self._t0 = time.time()
        return self._render(init=True), {"seed": seed, **_episode_info(st)}

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            action_str,
            prompt_format=self.cfg.prompt_format,
            action_sep=self.cfg.action_sep,
            max_actions=self.cfg.max_actions_per_step,
        )
        prims, all_valid = parse_primitives(parsed["actions"])
        before = self.sim.state()

        valid_actions: List[str] = []
        for prim in prims:
            if self.sim.done:
                break
            self.sim.execute(prim, self.cfg.max_move_mm, self.cfg.max_rotate_deg)
            valid_actions.append(str(prim))

        after = self.sim.state()
        success = bool(after["success"])
        action_ok = parsed["format_correct"] and all_valid and len(prims) > 0

        reward = 0.0
        if action_ok:
            reward += self.cfg.format_reward
        reward += self.cfg.progress_reward_per_mm * (before["trans_error_mm"] - after["trans_error_mm"])
        reward += self.cfg.progress_reward_per_deg * (before["angle_error_deg"] - after["angle_error_deg"])
        if success:
            reward += self.cfg.success_reward

        info: Dict[str, Any] = {k: v for k, v in parsed.items() if k != "llm_raw_response"}
        info.update(_episode_info(after))
        info["metrics"] = {
            "turn_metrics": {
                "action_is_valid": action_ok,
                "action_is_effective": len(valid_actions) > 0,
                "format_correct": parsed["format_correct"],
            },
            "traj_metrics": {"success": success},
        }
        info["episode_elapsed_seconds"] = time.time() - self._t0

        obs = self._render(init=False, valid_actions=valid_actions)
        return obs, float(reward), bool(self.sim.done), info

    # ------------------------------------------------------------------
    # Async interface (GymImageEnv)
    # ------------------------------------------------------------------

    async def _run(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.sim.executor, fn, *args)

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": system_prompt(
                format_name=self.cfg.prompt_format,
                max_actions_per_step=self.cfg.max_actions_per_step,
                action_sep=self.cfg.action_sep,
                max_move_mm=self.cfg.max_move_mm,
                max_rotate_deg=self.cfg.max_rotate_deg,
                trans_tol_mm=self.sim.trans_error * 10.0,
                angle_tol_deg=self.sim.angle_error,
                needs_jaw_close=self.sim.needs_jaw_close,
            )
        }

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await self._run(self._sync_reset, seed)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await self._run(self._sync_step, action_str)

    async def close(self) -> None:
        # The simulator is shared and long-lived; closing only frees the slot.
        if not self._closed:
            self._closed = True
            if self._on_close is not None:
                self._on_close()


def _episode_info(st: Dict[str, Any]) -> Dict[str, Any]:
    """Fields read by scripts/compare_results.py (same metrics as SurgicAI)."""
    return {
        "success": bool(st["success"]),
        "low_level_steps": int(st["low_level_steps"]),
        "traj_length_mm": float(st["traj_length_mm"]),
        "trans_error_mm": float(st["trans_error_mm"]),
        "angle_error_deg": float(st["angle_error_deg"]),
    }
