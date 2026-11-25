"""
SRCEnv: VAGEN GymImageEnv for SurgicAI's SRC suturing subtasks.

One VAGEN turn = one VLM response = up to `max_actions_per_step` mid-level
primitives, each expanded into SurgicAI low-level steps by SRCSim. An episode
ends on SurgicAI success or when the low-level step budget (300, as in
SurgicAI's evaluation) runs out; VAGEN's `max_turns` caps the turns.

All simulator calls run on SRCSim's single worker thread (AMBF / ROS handles
are not thread-safe), hence run_in_executor rather than asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

from vagen.envs import GymImageEnv

from .parse import VALID_FORMATS, parse_primitives, parse_response
from .prompt import (
    compact_observation_text,
    compact_system_prompt,
    get_format_instruction,
    observation_text,
    system_prompt,
)
from .sim import SRCSim
from .state_reward import JAW_OPEN_THRESHOLD, StateRewardConfig, instructions as state_reward_instructions, score_turn
from .expert_rag import ExpertIndex, default_expert_path
from .tasks import TASK_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)


@dataclass
class SRCEnvConfig:
    task: str = "approach"
    render_mode: str = "vision"         # vision | text
    prompt_format: str = "wm"           # wm | free_think | compact (offset in, action out)
    # Expert retrieval (expert_rag.py): show the action an expert demonstration took from
    # the most similar state as an "Expert hint" line. Needs prompt_format=compact.
    # Start every episode exactly where SurgicAI's expert demonstrations start (arm at its
    # initial pose, needle at its nominal pose, no needle noise). The seed then changes nothing.
    expert_start: bool = False
    expert_start_tol_mm: float = 1.0    # an arm start further than this from the expert's raises
    expert_start_tol_deg: float = 2.0
    # Also require the target (i.e. the needle) to be where the expert's was. Off by
    # default: in the old SRC scene a needle lying on the phantom cannot reproduce the
    # target recorded in SurgicAI's approach demonstrations (27 mm / 112 deg away, see
    # README), so the gap is only reported in the reset info.
    expert_start_check_target: bool = False
    rag_hint: bool = False
    rag_lookahead: int = 15             # expert low-level steps covered by one hint
    rag_expert_path: str = ""           # default: SurgicAI's Expert_traj/<Task>/all_episodes_merged.pkl
    # False: execute a salvageable <answer> even if the response is malformed
    # (it still earns no format reward). Same switch as VAGEN's own envs.
    strict_format: bool = False
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
    # World-model (state) reward, see state_reward.py. Per-turn reward for a perfect
    # <perception> / <prediction>; 0 switches that one off. Needs prompt_format=wm.
    state_estimation_reward: float = 0.0
    transition_prediction_reward: float = 0.0
    state_pos_full_mm: float = 0.5      # position error with full credit ...
    state_pos_zero_mm: float = 3.0      # ... and with none
    state_rot_full_deg: float = 1.0
    state_rot_zero_deg: float = 5.0
    state_score_base: float = 0.15
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
        self.state_reward = StateRewardConfig(
            state_estimation=self.cfg.state_estimation_reward,
            transition_prediction=self.cfg.transition_prediction_reward,
            pos_full_mm=self.cfg.state_pos_full_mm,
            pos_zero_mm=self.cfg.state_pos_zero_mm,
            rot_full_deg=self.cfg.state_rot_full_deg,
            rot_zero_deg=self.cfg.state_rot_zero_deg,
            score_base=self.cfg.state_score_base,
        )
        if self.state_reward.enabled and self.cfg.prompt_format != "wm":
            raise ValueError("state rewards score <perception>/<prediction>; set prompt_format: wm")
        self.expert = None
        if self.cfg.rag_hint:
            if self.cfg.prompt_format != "compact":
                raise ValueError("rag_hint adds a line to the compact observation; set prompt_format: compact")
            self.expert = _expert_index(
                self.cfg.rag_expert_path or default_expert_path(sim.task), self.cfg.rag_lookahead
            )
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
        if self.cfg.prompt_format == "compact":
            offset = [t - c for c, t in zip(st["gripper_pose"][:6], st["target_pose"][:6])]
            hint = ""
            if self.expert is not None:
                hint = str(self.expert.query(offset, st["jaw"], self.cfg.max_move_mm, self.cfg.max_rotate_deg))
            obs_str = compact_observation_text(
                offset=offset,
                jaw_open=st["jaw"] > JAW_OPEN_THRESHOLD,
                valid_actions=None if init else (valid_actions or []),
                expert_hint=hint,
                action_sep=self.cfg.action_sep,
            )
            if vision:
                obs_str = ph + "\n" + obs_str
            obs = {"obs_str": obs_str}
            if vision:
                obs["multi_modal_input"] = {ph: [self.sim.image()]}
            return obs
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
        st = self.sim.reset(seed, expert_start=self.cfg.expert_start)
        info: Dict[str, Any] = {"seed": seed, **_episode_info(st)}
        if self.cfg.expert_start:
            index = self.expert or _expert_index(
                self.cfg.rag_expert_path or default_expert_path(self.sim.task), self.cfg.rag_lookahead
            )
            gap = self.sim.start_error(index.first_state)
            info["expert_start_error"] = gap
            keys = ("gripper_pose", "target_pose") if self.cfg.expert_start_check_target else ("gripper_pose",)
            if (max(gap[f"{k}_mm"] for k in keys) > self.cfg.expert_start_tol_mm
                    or max(gap[f"{k}_deg"] for k in keys) > self.cfg.expert_start_tol_deg):
                raise RuntimeError(f"expert_start: episode does not start at the expert's start state: {gap}")
        self._t0 = time.time()
        return self._render(init=True), info

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            action_str,
            prompt_format=self.cfg.prompt_format,
            action_sep=self.cfg.action_sep,
            max_actions=self.cfg.max_actions_per_step,
            strict_format=self.cfg.strict_format,
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

        # What the agent said about the state, scored against the simulator. Gated on the
        # response format only: a turn can describe the state well and still act badly.
        state_scores = score_turn(
            {"perception": parsed.get("perception"), "prediction": parsed.get("prediction")},
            before,
            after,
            self.state_reward,
            format_correct=bool(parsed["format_correct"]),
        )
        reward += sum(state_scores.values())

        info: Dict[str, Any] = {k: v for k, v in parsed.items() if k != "llm_raw_response"}
        info.update(_episode_info(after))
        info.update({f"state_reward/{name}": float(v) for name, v in state_scores.items()})
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.sim.executor, fn, *args)

    async def system_prompt(self) -> Dict[str, Any]:
        if self.cfg.prompt_format == "compact":
            return {
                "obs_str": compact_system_prompt(
                    instruction=TASK_INSTRUCTIONS[self.sim.task],
                    max_actions_per_step=self.cfg.max_actions_per_step,
                    action_sep=self.cfg.action_sep,
                    max_move_mm=self.cfg.max_move_mm,
                    max_rotate_deg=self.cfg.max_rotate_deg,
                    trans_tol_mm=self.sim.trans_error * 10.0,
                    angle_tol_deg=self.sim.angle_error,
                    needs_jaw_close=self.sim.needs_jaw_close,
                    expert_hint=self.expert is not None,
                )
            }
        extra = state_reward_instructions(tuple(self.state_reward.enabled))
        return {
            "obs_str": system_prompt(
                extra_instructions=extra,
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


_EXPERT_CACHE: Dict[Tuple[str, int], ExpertIndex] = {}


def _expert_index(path: str, lookahead: int) -> ExpertIndex:
    """Environments are created per episode; load each expert file once per server."""
    key = (path, int(lookahead))
    if key not in _EXPERT_CACHE:
        _EXPERT_CACHE[key] = ExpertIndex.load(path, lookahead=lookahead)
        LOGGER.info("expert index: %d states / %d episodes from %s", len(_EXPERT_CACHE[key]),
                    _EXPERT_CACHE[key].n_episodes, path)
    return _EXPERT_CACHE[key]


def _episode_info(st: Dict[str, Any]) -> Dict[str, Any]:
    """Fields read by scripts/compare_results.py (same metrics as SurgicAI)."""
    return {
        "success": bool(st["success"]),
        "low_level_steps": int(st["low_level_steps"]),
        "traj_length_mm": float(st["traj_length_mm"]),
        "trans_error_mm": float(st["trans_error_mm"]),
        "angle_error_deg": float(st["angle_error_deg"]),
    }
