"""
World-model (state) reward for SRC: score what the agent *says* about the state.

VAGEN's world-modeling RL pays the agent for two descriptions besides the task reward:

* state estimation      -- ``<perception>``: the state it acts *from*;
* transition prediction -- ``<prediction>``: the state it acts *into*.

VAGEN's own implementation (``vagen/envs/_common/rewards``) wraps a locally built
environment, reads categorical relations ("above / left of the player") out of it and
has an LLM judge structure the agent's sentences. Neither half fits here: the SRC
environment lives behind ``RemoteEnv`` (the true state is on the simulator machine) and
the state is a metric 6-DoF pose, for which the VAGEN paper itself reports that a
structured format is needed in manipulation tasks. So the scoring is done on the
simulator side, inside ``SRCEnv.step``, against the simulator's own numbers, with a
closed-vocabulary sentence and a deterministic parser -- no judge model.

What the agent describes (the same sentence in both sections)::

    offset position (dx, dy, dz) mm, orientation (droll, dpitch, dyaw) deg, jaw open

* ``offset`` is target minus gripper, per axis, in the units and frame of the
  observation text -- exactly the quantity a correct ``move`` / ``rotate`` consumes;
* ``jaw`` is ``open`` or ``closed``.

``<perception>`` is checked against the state before the turn's actions,
``<prediction>`` against the state after they were executed (so clipping of over-long
moves and an early gripper close are things the agent has to predict).

Score of one description, in [0, 1]:

* position, orientation and jaw are three groups weighted ``GROUP_WEIGHTS``;
* each number earns 1 inside ``full`` error, 0 beyond ``zero`` error, linear between;
  a group is the mean of its three numbers; a group that is not stated scores 0;
* ``score_base`` is subtracted and the rest rescaled (as VAGEN does), because a
  description that looked at nothing already scores something.

Calibration (10 recorded approach start states, seeds 20000-20009, 2026-11-13). The
bands are tight on purpose -- the observation prints every number to 0.1, so the offset
is an exact subtraction -- and they are what keeps constant answers cheap:

* "all offsets zero, jaw open" scores 0.15 on average (max 0.27): the default
  ``score_base``. With looser bands (1 -> 10 mm, 2 -> 20 deg) it scored 0.40.
* the best constant for the FIRST turn (the mean start offset; the gripper always starts
  in the same place and only the needle moves) still scores 0.46 on average, i.e. about
  a third of that turn's reward after ``score_base``. It is worthless on later turns,
  whose offsets depend on the actions taken. Ten states is a small sample.

Like VAGEN, format is a gate rather than a line item: a turn whose response is not the
canonical WM format, or that leaves out a requested section's position/orientation,
earns no state reward at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

#: reward name -> the response section it scores (VAGEN's names)
TAGS = {"state_estimation": "perception", "transition_prediction": "prediction"}

GROUP_WEIGHTS = {"position": 0.45, "orientation": 0.45, "jaw": 0.1}

#: Jaw angle above which the jaw counts as open (SurgicAI: 0.8 open, 0.0 closed).
JAW_OPEN_THRESHOLD = 0.4

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"
_TRIPLE = r"\(\s*({n})\s*,\s*({n})\s*,\s*({n})\s*\)".format(n=_NUM)
_POSITION_RE = re.compile(r"position\s*:?\s*" + _TRIPLE, re.IGNORECASE)
_ORIENTATION_RE = re.compile(r"orientation\s*:?\s*" + _TRIPLE, re.IGNORECASE)
_JAW_RE = re.compile(r"jaw\s*:?\s*(?:is\s+)?(open|closed|close)\b", re.IGNORECASE)

SENTENCE = "offset position (dx, dy, dz) mm, orientation (droll, dpitch, dyaw) deg, jaw open"


@dataclass
class StateRewardConfig:
    """Per-turn reward for a perfect description of each section (0 switches it off)."""

    state_estimation: float = 0.0
    transition_prediction: float = 0.0
    pos_full_mm: float = 0.5
    pos_zero_mm: float = 3.0
    rot_full_deg: float = 1.0
    rot_zero_deg: float = 5.0
    score_base: float = 0.15

    @property
    def enabled(self) -> Dict[str, float]:
        both = {"state_estimation": self.state_estimation, "transition_prediction": self.transition_prediction}
        return {name: float(v) for name, v in both.items() if v and float(v) > 0.0}


def true_state(sim_state: Dict[str, Any]) -> Dict[str, Any]:
    """The description a perfect agent would give of `SRCSim.state()`."""
    cur, tgt = sim_state["gripper_pose"], sim_state["target_pose"]
    offset = [float(t) - float(c) for c, t in zip(cur[:6], tgt[:6])]
    return {
        "position": offset[0:3],
        "orientation": offset[3:6],
        "jaw": "open" if float(sim_state["jaw"]) > JAW_OPEN_THRESHOLD else "closed",
    }


def format_state(state: Dict[str, Any]) -> str:
    """Render a state in the sentence the parser reads (used by prompts and tests)."""
    p, o = state["position"], state["orientation"]
    return (
        f"offset position ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm, "
        f"orientation ({o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}) deg, jaw {state['jaw']}"
    )


def parse_state(text: Optional[str]) -> Dict[str, Any]:
    """Read one description. Groups that are not stated stay None and earn nothing."""
    text = text or ""
    out: Dict[str, Any] = {"position": None, "orientation": None, "jaw": None}
    for key, regex in (("position", _POSITION_RE), ("orientation", _ORIENTATION_RE)):
        m = regex.search(text)
        if m:
            out[key] = [float(v) for v in m.groups()]
    m = _JAW_RE.search(text)
    if m:
        out["jaw"] = "open" if m.group(1).lower() == "open" else "closed"
    return out


def _closeness(pred: Sequence[float], gold: Sequence[float], full: float, zero: float) -> float:
    total = 0.0
    for p, g in zip(pred, gold):
        err = abs(float(p) - float(g))
        if err <= full:
            total += 1.0
        elif err < zero:
            total += (zero - err) / (zero - full)
    return total / len(gold)


def description_score(pred: Dict[str, Any], gold: Dict[str, Any], cfg: StateRewardConfig) -> float:
    """Raw quality of one description in [0, 1], before `score_base`."""
    score = 0.0
    if pred.get("position") is not None:
        score += GROUP_WEIGHTS["position"] * _closeness(
            pred["position"], gold["position"], cfg.pos_full_mm, cfg.pos_zero_mm
        )
    if pred.get("orientation") is not None:
        score += GROUP_WEIGHTS["orientation"] * _closeness(
            pred["orientation"], gold["orientation"], cfg.rot_full_deg, cfg.rot_zero_deg
        )
    if pred.get("jaw") is not None and pred["jaw"] == gold["jaw"]:
        score += GROUP_WEIGHTS["jaw"]
    return score


def _above_base(score: float, base: float) -> float:
    if base <= 0.0:
        return score
    return max(0.0, (score - base) / (1.0 - base))


def score_turn(
    sections: Dict[str, Optional[str]],
    before: Dict[str, Any],
    after: Dict[str, Any],
    cfg: StateRewardConfig,
    format_correct: bool,
) -> Dict[str, float]:
    """Reward earned by this turn's descriptions, per enabled reward name.

    `sections` maps the section tag ("perception" / "prediction") to its text; `before`
    and `after` are `SRCSim.state()` around the turn's actions.
    """
    enabled = cfg.enabled
    rewards = {name: 0.0 for name in enabled}
    if not enabled or not format_correct:
        return rewards

    parsed = {name: parse_state(sections.get(TAGS[name])) for name in enabled}
    # A requested section that states no position or no orientation voids the turn:
    # paying per section would make the other one optional.
    if any(p["position"] is None or p["orientation"] is None for p in parsed.values()):
        return rewards

    gold = {"state_estimation": true_state(before), "transition_prediction": true_state(after)}
    for name, scale in enabled.items():
        quality = _above_base(description_score(parsed[name], gold[name], cfg), cfg.score_base)
        rewards[name] = scale * quality
    return rewards


def instructions(enabled: Sequence[str]) -> str:
    """System-prompt block telling the agent what the scored sections must contain."""
    if not enabled:
        return ""
    lines: List[str] = ["State descriptions (each is checked against the simulator and earns reward when accurate):"]
    if "state_estimation" in enabled:
        lines.append(
            "- <perception> must give the CURRENT offset from the gripper to the target "
            "(target minus gripper, for each of the six numbers) and the jaw state."
        )
    if "transition_prediction" in enabled:
        lines.append(
            "- <prediction> must give the offset and jaw state that will hold AFTER your action(s) "
            "are executed (values beyond the per-action limits are clipped, so the offset may not reach zero)."
        )
    lines.append(f"Write each of them exactly in this form:\n  {SENTENCE}")
    lines.append("with 'jaw open' or 'jaw closed'. For the example above:")
    if "state_estimation" in enabled:
        lines.append(
            "<perception>offset position (8.0, 5.5, -10.0) mm, orientation (5.0, -3.0, 5.0) deg, jaw open</perception>"
        )
    if "transition_prediction" in enabled:
        lines.append(
            "<prediction>offset position (0.0, 0.0, 0.0) mm, orientation (0.0, 0.0, 0.0) deg, jaw open</prediction>"
        )
    return "\n".join(lines)
