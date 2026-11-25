"""
Prompt templates for the SRC suturing environment.
"""

from __future__ import annotations

from typing import List, Optional

_FORMAT_INSTRUCTIONS = {
    "free_think": (
        "You need to think first, then give your answer. Respond in this format:\n"
        "<think>...</think><answer>{action_example}</answer>"
    ),
    "wm": (
        "Describe your perception, explain your reasoning, predict the next state, and "
        "then give the action. Respond in this exact order:\n"
        "<perception>...</perception><reasoning>...</reasoning>"
        "<prediction>...</prediction><answer>{action_example}</answer>"
    ),
}

_ROBOT_DESCRIPTION = """\
You are controlling a simulated da Vinci surgical robot (Surgical Robotics Challenge) performing a suturing subtask. \
There are two patient-side manipulators: PSM1 (left arm) and PSM2 (right arm). Each turn you command the ACTIVE arm only.

Action Space Guide
1. move(dx, dy, dz)             # translate the active gripper by (dx, dy, dz) millimetres in its robot base frame
2. rotate(droll, dpitch, dyaw)  # rotate the active gripper by roll/pitch/yaw increments in degrees
3. gripper(open) / gripper(close)

Hints:
1. Each move is limited to {max_move_mm} mm per axis and each rotate to {max_rotate_deg} degrees per axis; larger values are clipped.
2. Poses are reported as (x, y, z) in millimetres and (roll, pitch, yaw) in degrees, in the active arm's base frame.
3. Use large moves when far from the target and small corrections when close.
4. The task succeeds when the gripper is within {trans_tol_mm:g} mm and {angle_tol_deg:g} degrees of the target{jaw_hint}.

Example:
Gripper pose: position (12.0, -30.5, -85.0) mm, orientation (-200.0, 3.0, 110.0) deg, jaw 0.80
Target pose:  position (20.0, -25.0, -95.0) mm, orientation (-195.0, 0.0, 115.0) deg
Answer: move(8, 5.5, -10)|rotate(5, -3, 5)"""


# Compact prompt: the observation is the offset to the target (the number the action
# consumes) and the reply is the action alone. The no_concat harness sends every turn as
# its own conversation, so nothing of an earlier reply is carried over: the only part of
# a reply that reaches the next turn is the executed <answer>, echoed as "Last action".
_COMPACT_DESCRIPTION = """\
You control the active arm of a simulated da Vinci surgical robot. Task: {instruction}
Each turn you get the offset from the gripper to the target (target minus gripper). Output actions that cancel it.
Actions:
move(dx, dy, dz)             # millimetres, at most {max_move_mm:g} per axis
rotate(droll, dpitch, dyaw)  # roll/pitch/yaw in degrees, at most {max_rotate_deg:g} per axis
gripper(open) / gripper(close)
Up to {max_actions} actions per turn, separated by '{action_sep}'. Larger values are clipped, so a far target takes several turns.
Success: within {trans_tol_mm:g} mm and {angle_tol_deg:g} deg of the target{jaw_hint}.{close_hint}{hint_line}
Reply with only <answer>...</answer>.

Example:
Offset: position (8.0, 5.5, -10.0) mm, orientation (5.0, -3.0, 5.0) deg, jaw open
<answer>move(8.0, 5.5, -10.0){action_sep}rotate(5.0, -3.0, 5.0){example_tail}</answer>"""


def compact_system_prompt(
    instruction: str,
    max_actions_per_step: int,
    action_sep: str,
    max_move_mm: float,
    max_rotate_deg: float,
    trans_tol_mm: float,
    angle_tol_deg: float,
    needs_jaw_close: bool,
    expert_hint: bool = False,
) -> str:
    return _COMPACT_DESCRIPTION.format(
        instruction=instruction,
        max_move_mm=max_move_mm,
        max_rotate_deg=max_rotate_deg,
        max_actions=max_actions_per_step,
        action_sep=action_sep,
        trans_tol_mm=trans_tol_mm,
        angle_tol_deg=angle_tol_deg,
        jaw_hint=" with the jaw closed" if needs_jaw_close else "",
        close_hint=" Close the gripper in the turn whose motion reaches the target." if needs_jaw_close else "",
        hint_line=(
            "\nAn 'Expert hint' line gives the action an expert demonstration took from the most similar state."
            if expert_hint else ""
        ),
        example_tail=f"{action_sep}gripper(close)" if needs_jaw_close else "",
    )


def compact_observation_text(
    offset: List[float],
    jaw_open: bool,
    valid_actions: Optional[List[str]] = None,
    expert_hint: str = "",
    action_sep: str = "|",
) -> str:
    dx, dy, dz, dr, dp, dyw = offset[:6]
    lines = []
    if valid_actions is not None:
        lines.append(f"Last action: {action_sep.join(valid_actions) if valid_actions else 'none (invalid answer)'}")
    lines.append(
        f"Offset: position ({dx:.1f}, {dy:.1f}, {dz:.1f}) mm, "
        f"orientation ({dr:.1f}, {dp:.1f}, {dyw:.1f}) deg, jaw {'open' if jaw_open else 'closed'}"
    )
    if expert_hint:
        lines.append(f"Expert hint: {expert_hint}")
    return "\n".join(lines)


def get_format_instruction(format_name: str, max_actions_per_step: int, action_sep: str) -> str:
    if format_name not in _FORMAT_INSTRUCTIONS:
        raise ValueError(f"Unknown format {format_name!r}. Available: {sorted(_FORMAT_INSTRUCTIONS)}")
    action_example = f"action1{action_sep} action2{action_sep} ..."
    return (
        f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.\n"
        + _FORMAT_INSTRUCTIONS[format_name].format(action_example=action_example)
    )


def system_prompt(
    format_name: str,
    max_actions_per_step: int,
    action_sep: str,
    max_move_mm: float,
    max_rotate_deg: float,
    trans_tol_mm: float,
    angle_tol_deg: float,
    needs_jaw_close: bool,
    extra_instructions: str = "",
) -> str:
    desc = _ROBOT_DESCRIPTION.format(
        max_move_mm=max_move_mm,
        max_rotate_deg=max_rotate_deg,
        trans_tol_mm=trans_tol_mm,
        angle_tol_deg=angle_tol_deg,
        jaw_hint=" with the jaw closed" if needs_jaw_close else "",
    )
    text = desc + "\n\n" + get_format_instruction(format_name, max_actions_per_step, action_sep)
    if extra_instructions:
        text += "\n\n" + extra_instructions
    return text


def _fmt_pose(pose_mm_deg: List[float]) -> str:
    x, y, z, r, p, yw = pose_mm_deg[:6]
    return f"position ({x:.1f}, {y:.1f}, {z:.1f}) mm, orientation ({r:.1f}, {p:.1f}, {yw:.1f}) deg"


def observation_text(
    init: bool,
    image_placeholder: Optional[str],
    instruction: str,
    active_arm: str,
    gripper_pose: List[float],
    jaw: float,
    target_pose: Optional[List[float]],
    steps_left: int,
    valid_actions: Optional[List[str]] = None,
    format_prompt: str = "",
) -> str:
    lines = []
    if init:
        lines.append("[Initial Observation]:")
    else:
        lines.append(f"After your answer, the extracted valid action(s) is {valid_actions or []}.")
        lines.append("After that, the observation is:")
    if image_placeholder:
        lines.append(image_placeholder)
    lines.append(f"Task: {instruction}")
    lines.append(f"Active arm: {active_arm}")
    lines.append(f"Gripper pose: {_fmt_pose(gripper_pose)}, jaw {jaw:.2f}")
    if target_pose is not None:
        lines.append(f"Target pose:  {_fmt_pose(target_pose)}")
    lines.append(f"Remaining low-level step budget: {steps_left}")
    lines.append("Decide your next action(s).")
    text = "\n".join(lines)
    if format_prompt:
        text += "\n" + format_prompt
    return text
