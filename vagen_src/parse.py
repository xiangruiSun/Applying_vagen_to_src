"""
Parsing for VLM responses and mid-level SRC primitives.

Response formats (same as VAGEN's primitive_skill env):
  - free_think: <think>...</think><answer>...</answer>
  - wm:         <observation>...</observation><think>...</think>
                <answer>...</answer><prediction>...</prediction>

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

_PARSE_PATTERNS = {
    "free_think": r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    "wm": (
        r"<observation>(.*?)</observation>\s*"
        r"<think>(.*?)</think>\s*"
        r"<answer>(.*?)</answer>\s*"
        r"<prediction>(.*?)</prediction>"
    ),
}

VALID_FORMATS = list(_PARSE_PATTERNS.keys())

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
) -> Dict[str, Any]:
    """Extract reasoning fields and the raw action strings from an LLM response."""
    result: Dict[str, Any] = {
        "llm_raw_response": response,
        "actions": [],
        "format_correct": False,
    }
    pattern = _PARSE_PATTERNS.get(prompt_format)
    if pattern is None:
        raise ValueError(f"Unknown prompt_format: {prompt_format}. Valid: {VALID_FORMATS}")

    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return result

    if prompt_format == "free_think":
        result["think"] = match.group(1).strip()
        action_text = match.group(2).strip()
    else:
        result["observation"] = match.group(1).strip()
        result["think"] = match.group(2).strip()
        action_text = match.group(3).strip()
        result["prediction"] = match.group(4).strip()

    result["format_correct"] = True
    result["actions"] = [a.strip() for a in action_text.split(action_sep) if a.strip()][:max_actions]
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
