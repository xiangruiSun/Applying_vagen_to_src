"""
Parsing for VLM responses and mid-level SRC primitives.

Response formats (VAGEN's shared protocol, vagen/envs/_common/response_format.py):
  - free_think: <think>...</think><answer>...</answer>
  - wm:         <perception>...</perception><reasoning>...</reasoning>
                <prediction>...</prediction><answer>...</answer>

Mid-level primitives inside <answer>, separated by `action_sep`:
  move(dx, dy, dz)            translate the active gripper, millimetres, robot base frame
  rotate(droll, dpitch, dyaw) rotate the active gripper, degrees (RPY increments)
  gripper(open|close)         open / close the jaw

Each primitive is expanded by SRCSim into SurgicAI's low-level 7-D actions,
so a VLM turn covers many simulator steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

VALID_FORMATS = ["free_think", "wm"]

_PRIMITIVE_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass
class Primitive:
    name: str                       # move | rotate | gripper
    values: Tuple[float, ...] = ()  # mm for move, degrees for rotate
    jaw_open: Optional[bool] = None  # for gripper

    def __str__(self) -> str:
        if self.name == "gripper":
            return f"gripper({'open' if self.jaw_open else 'close'})"
        return f"{self.name}({', '.join(f'{v:g}' for v in self.values)})"


def parse_response(
    response: str,
    prompt_format: str = "wm",
    action_sep: str = "|",
    max_actions: int = 3,
    strict_format: bool = False,
) -> Dict[str, Any]:
    """Extract reasoning fields and the raw action strings from an LLM response.

    `format_correct` is VAGEN's strict protocol bit (earns the format reward).
    With strict_format=False a malformed response whose <answer> can still be
    found is executed anyway (no format reward), as VAGEN's own eval configs do.
    """
    from vagen.envs._common.response_format import (
        action_count_within_limit,
        parse_free_think_sections,
        parse_wm_sections,
        split_actions,
    )

    if prompt_format == "wm":
        sec = parse_wm_sections(response)
    elif prompt_format == "free_think":
        sec = parse_free_think_sections(response)
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}. Valid: {VALID_FORMATS}")

    format_correct = bool(sec.format_correct) and action_count_within_limit(sec.answer, action_sep, max_actions)
    result: Dict[str, Any] = {
        "llm_raw_response": response,
        "perception": sec.perception,
        "reasoning": sec.reasoning,
        "prediction": sec.prediction,
        "actions": [],
        "format_correct": format_correct,
    }
    if format_correct or (not strict_format and sec.answer):
        result["actions"] = split_actions(sec.answer, action_sep, max_actions)
    return result


def parse_primitive(action_str: str) -> Optional[Primitive]:
    """Parse one primitive string; return None if it is malformed."""
    m = _PRIMITIVE_RE.match(action_str)
    if not m:
        return None
    name = m.group(1).strip().lower()
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]

    if name == "gripper":
        if len(args) != 1 or args[0].lower() not in ("open", "close"):
            return None
        return Primitive(name="gripper", jaw_open=(args[0].lower() == "open"))

    if name in ("move", "rotate"):
        if len(args) != 3:
            return None
        try:
            values = tuple(float(a) for a in args)
        except ValueError:
            return None
        return Primitive(name=name, values=values)

    return None


def parse_primitives(actions: List[str]) -> Tuple[List[Primitive], bool]:
    """Parse action strings in order, stopping at the first invalid one.

    Returns (primitives, all_valid).
    """
    prims: List[Primitive] = []
    for a in actions:
        p = parse_primitive(a)
        if p is None:
            return prims, False
        prims.append(p)
    return prims, True
