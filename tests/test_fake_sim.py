"""
Offline test of the glue code with a fake SurgicAI env (no ROS / AMBF needed).

Checks primitive parsing, primitive -> low-level step expansion, SRCEnv's
VAGEN interface and the full HTTP path through VAGEN's GymService and
GymImageEnvClient. Run with the simulator-side Python:

    pip install pytest numpy pillow fastapi "uvicorn<0.41" httpx
    python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import socket
import sys
import threading
import time
import types

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from vagen_src.paths import setup_vagen_path  # noqa: E402

setup_vagen_path()

from vagen_src.parse import parse_primitive, parse_response  # noqa: E402


class FakeApproachEnv:
    """Mimics the parts of SurgicAI's SRC_approach that SRCSim relies on."""

    def __init__(self, seed=None, render_mode=None, reward_type="sparse", threshold=(0.5, 0.5),
                 max_episode_step=300, step_size=None):
        self.step_size = np.asarray(step_size, dtype=np.float32)
        self.threshold_trans, self.threshold_angle = float(threshold[0]), float(threshold[1])
        self.psm_idx = 2
        self.psm_goal_list = [np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)]
        self.random_range = np.array([0.02, 0.02, 0.3], dtype=np.float32)

    def reset(self, **kwargs):
        start = np.array([0.0, 0.0, -0.08, -3.0, 0.0, 1.0, 0.8], dtype=np.float32)
        r = self.random_range
        noise = np.r_[np.random.uniform(-r[0], r[0], 3), np.random.uniform(-r[2], r[2], 3), 0.0]
        self.psm_goal_list[1] = start
        self.goal_obs = (start + np.r_[0.01, 0.0, -0.02, 0.2, 0.0, -0.1, 0.0] + noise).astype(np.float32)
        self.goal_obs[6] = 0.0
        self._update()
        return self.obs, {"is_success": False}

    def step(self, action):
        self.psm_goal_list[1] = self.psm_goal_list[1] + action * self.step_size
        self._update()
        cur, goal = self.obs["achieved_goal"], self.obs["desired_goal"]
        ok = (np.linalg.norm(cur[:3] - goal[:3]) <= self.threshold_trans
              and np.linalg.norm(cur[3:6] - goal[3:6]) <= self.threshold_angle
              and self.psm_goal_list[1][6] <= 0.1)
        return self.obs, 0.0 if ok else -1.0, bool(ok), False, {"is_success": bool(ok)}

    def _update(self):
        scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)  # m -> cm like SurgicAI
        cur, goal = self.psm_goal_list[1] * scale, self.goal_obs * scale
        self.obs = {"observation": np.r_[cur, goal, goal - cur], "achieved_goal": cur, "desired_goal": goal}


class FakeCamera:
    def grab(self, timeout=5.0, newer_than=0.0):
        return Image.new("RGB", (64, 48), (200, 80, 80))


@pytest.fixture(autouse=True)
def fake_surgicai(monkeypatch):
    mod = types.ModuleType("Approach_env")
    mod.SRC_approach = FakeApproachEnv
    monkeypatch.setitem(sys.modules, "Approach_env", mod)


def _make_sim():
    from vagen_src.sim import SRCSim

    sim = SRCSim(task="approach", camera_topic=None)
    sim.camera = FakeCamera()
    return sim


def _oracle():
    spec = importlib.util.spec_from_file_location("smoke", os.path.join(REPO_ROOT, "scripts", "smoke_test_env.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.oracle_answer


CFG = {"task": "approach", "render_mode": "vision", "prompt_format": "wm"}


def test_parsing():
    r = parse_response("<perception>a</perception><reasoning>b</reasoning><prediction>c</prediction>"
                       "<answer>move(1,2,3)|gripper(close)</answer>", "wm")
    assert r["format_correct"] and r["actions"] == ["move(1,2,3)", "gripper(close)"]
    assert parse_primitive("rotate(5, -3, 0)").values == (5.0, -3.0, 0.0)
    assert parse_primitive("gripper(open)").jaw_open is True
    assert parse_primitive("move(1,2)") is None
    assert parse_primitive("fly(1,2,3)") is None
    assert not parse_response("move(1,2,3)", "wm")["format_correct"]
    # Malformed but salvageable: executed unless strict, never format-correct.
    loose = "I think <answer>move(1,2,3)</answer>"
    assert parse_response(loose, "wm")["actions"] == ["move(1,2,3)"]
    assert not parse_response(loose, "wm")["format_correct"]
    assert parse_response(loose, "wm", strict_format=True)["actions"] == []


def test_primitive_expansion_counts_steps_like_surgicai():
    from vagen_src.parse import Primitive

    sim = _make_sim()
    sim.reset(0)
    start = sim.env.psm_goal_list[1].copy()
    n = sim.execute(Primitive("move", (10.0, -4.0, 0.0)), 20, 30)
    assert n == 10                                   # 10 mm at 1 mm / step
    assert np.allclose(sim.env.psm_goal_list[1][:3] - start[:3], [0.010, -0.004, 0.0], atol=1e-6)
    assert sim.traj_length_mm == pytest.approx(np.hypot(10, 4), rel=1e-4)
    assert sim.execute(Primitive("move", (100.0, 0.0, 0.0)), 20, 30) == 20  # clipped to 20 mm


def test_seeded_resets_are_reproducible():
    sim = _make_sim()
    a = sim.reset(123)["target_pose"]
    sim.reset(7)
    assert sim.reset(123)["target_pose"] == a


def test_env_episode_reaches_success():
    from vagen_src.src_env import SRCEnv

    oracle = _oracle()

    async def run():
        env = SRCEnv(CFG, sim=_make_sim())
        assert "move(dx, dy, dz)" in (await env.system_prompt())["obs_str"]
        obs, info = await env.reset(5)
        assert obs["obs_str"].count("<image>") == len(obs["multi_modal_input"]["<image>"]) == 1
        total, done, turns = 0.0, False, 0
        while not done and turns < 12:
            obs, reward, done, info = await env.step(oracle(obs["obs_str"], 20, 30))
            total += reward
            turns += 1
        await env.close()
        return info, total

    info, total = asyncio.run(run())
    assert info["success"] and info["metrics"]["traj_metrics"]["success"]
    assert 0 < info["low_level_steps"] <= 300
    assert total > 10


def test_invalid_answer_gets_no_format_reward():
    from vagen_src.src_env import SRCEnv

    async def run():
        env = SRCEnv(CFG, sim=_make_sim())
        await env.reset(1)
        return await env.step("I would move left")

    _, reward, done, info = asyncio.run(run())
    assert reward == 0.0 and not done and info["low_level_steps"] == 0


def test_state_reward_parsing_and_scoring():
    from vagen_src.state_reward import StateRewardConfig, description_score, format_state, parse_state, true_state

    st = {"gripper_pose": [-37.2, 12.1, -80.4, -154.9, 4.4, 116.7], "target_pose": [-31.6, -7.4, -102.8, -222.5, 28.1, 69.0], "jaw": 0.8}
    gold = true_state(st)
    assert gold["jaw"] == "open" and gold["position"] == pytest.approx([5.6, -19.5, -22.4])
    assert gold["orientation"] == pytest.approx([-67.6, 23.7, -47.7])

    cfg = StateRewardConfig(state_estimation=1.0)
    assert parse_state(format_state(gold)) == {k: pytest.approx(v) if k != "jaw" else v for k, v in gold.items()}
    assert description_score(parse_state(format_state(gold)), gold, cfg) == pytest.approx(1.0)
    # Free text states nothing scoreable.
    vague = parse_state("PSM2's gripper is open and positioned to grasp the needle.")
    assert vague == {"position": None, "orientation": None, "jaw": None}
    assert description_score(vague, gold, cfg) == 0.0
    # x right, y 1.5 mm off (partial), z far off; orientation missing; jaw wrong.
    partial = parse_state("offset position (5.6, -18.0, 0) mm, jaw closed")
    assert description_score(partial, gold, cfg) == pytest.approx(0.45 * (1.0 + 1.5 / 2.5 + 0.0) / 3.0)


def test_state_reward_in_env_step():
    from vagen_src.src_env import SRCEnv
    from vagen_src.state_reward import format_state, true_state

    cfg = dict(CFG, state_estimation_reward=0.2, transition_prediction_reward=0.3, progress_reward_per_mm=0.0,
               progress_reward_per_deg=0.0, format_reward=0.0)

    def answer(perception, prediction, action="move(1, 0, 0)"):
        return (f"<perception>{perception}</perception><reasoning>r</reasoning>"
                f"<prediction>{prediction}</prediction><answer>{action}</answer>")

    async def run():
        sim = _make_sim()
        env = SRCEnv(cfg, sim=sim)
        prompt = (await env.system_prompt())["obs_str"]
        assert "<perception>offset position" in prompt and "<prediction>offset position" in prompt
        await env.reset(3)
        before = true_state(sim.state())
        after = dict(before, position=[before["position"][0] - 1.0, *before["position"][1:]])
        out = []
        # 1. both descriptions exact  2. exact perception, free-text prediction
        # 3. malformed response carrying an exact perception
        out.append(await env.step(answer(format_state(before), format_state(after))))
        b2 = true_state(sim.state())
        out.append(await env.step(answer(format_state(b2), "the gripper will be closer")))
        b3 = true_state(sim.state())
        out.append(await env.step(f"<perception>{format_state(b3)}</perception><answer>move(1, 0, 0)</answer>"))
        await env.close()
        return out

    (_, r1, _, i1), (_, r2, _, i2), (_, r3, _, i3) = asyncio.run(run())
    assert r1 == pytest.approx(0.5) and i1["state_reward/state_estimation"] == pytest.approx(0.2)
    assert i1["state_reward/transition_prediction"] == pytest.approx(0.3)
    assert r2 == 0.0 and i2["state_reward/state_estimation"] == 0.0      # a missing section voids the turn
    assert r3 == 0.0 and i3["low_level_steps"] == 3                        # malformed: action runs, no state reward


def test_state_reward_off_by_default():
    from vagen_src.src_env import SRCEnv

    async def run():
        env = SRCEnv(CFG, sim=_make_sim())
        prompt = (await env.system_prompt())["obs_str"]
        await env.reset(1)
        return prompt, await env.step("<perception>p</perception><reasoning>r</reasoning><prediction>q</prediction><answer>gripper(open)</answer>")

    prompt, (_, reward, _, info) = asyncio.run(run())
    assert "State descriptions" not in prompt and not any(k.startswith("state_reward/") for k in info)
    assert reward == pytest.approx(0.1)


def test_remote_roundtrip_through_vagen_service():
    import uvicorn
    from vagen.envs._common.remote.service import GymService
    from vagen.envs.remote import GymImageEnvClient

    from vagen_src.handler import SRCHandler

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    app = GymService(SRCHandler(_make_sim())).build()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    oracle = _oracle()

    async def run():
        results = []
        for seed in (11, 12):  # second episode must get the slot after the first closes
            env = GymImageEnvClient({"base_urls": f"http://127.0.0.1:{port}", "timeout": 30, "retries": 2, **CFG})
            obs, _ = await env.reset(seed)
            assert len(obs["multi_modal_input"]["<image>"]) == 1
            done, info = False, {}
            for _ in range(12):
                obs, _, done, info = await env.step(oracle(obs["obs_str"], 20, 30))
                if done:
                    break
            await env.close()
            results.append(info)
        return results

    try:
        results = asyncio.run(run())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert all(r["success"] for r in results)


# ----------------------------------------------------------------------
# Compact prompt (offset in, action out) and expert retrieval
# ----------------------------------------------------------------------

COMPACT = {"task": "approach", "render_mode": "text", "prompt_format": "compact"}
_OFFSET_RE = re.compile(r"Offset: position \(([^)]*)\) mm, orientation \(([^)]*)\) deg, jaw (open|closed)")


def _fake_expert(n_episodes=3, steps=40):
    """Straight-line demonstrations to one goal, then a jaw close, in SurgicAI's units (cm, rad)."""
    goal = np.array([-6.0, 1.0, -12.0, -3.9, -0.1, 3.0, 0.0])
    out = []
    for e in range(n_episodes):
        start = goal + np.array([2.2, 0.3, 4.0, 1.2, 0.2, -1.0, 0.8]) + 0.01 * e
        path = [start + (goal - start) * np.r_[np.full(6, min(1.0, t / (steps - 10))), 0.0] for t in range(steps - 9)]
        for jaw in np.linspace(0.7, 0.0, 9):
            path.append(np.r_[path[-1][:6], jaw])
        for t in range(len(path) - 1):
            out.append({
                "obs": {"achieved_goal": path[t], "desired_goal": goal},
                "next_obs": {"achieved_goal": path[t + 1], "desired_goal": goal},
                "action": np.zeros(7), "reward": np.zeros(1),
                "done": np.array([float(t == len(path) - 2)]), "info": {"is_success": t == len(path) - 2},
            })
    return out


def test_compact_format_parsing():
    r = parse_response("<answer>move(1,2,3)|gripper(close)</answer>", "compact")
    assert r["format_correct"] and r["actions"] == ["move(1,2,3)", "gripper(close)"]
    assert parse_response("<think>\n\n</think>\n<answer>move(1,2,3)</answer>", "compact")["format_correct"]
    chatty = parse_response("Sure! <answer>move(1,2,3)</answer>", "compact")
    assert not chatty["format_correct"] and chatty["actions"] == ["move(1,2,3)"]


def test_compact_observation_is_the_offset_in_rpy_and_short():
    from vagen_src.src_env import SRCEnv

    async def run():
        sim = _make_sim()
        env = SRCEnv(COMPACT, sim=sim)
        wm = SRCEnv({**COMPACT, "prompt_format": "wm"}, sim=sim)
        obs, _ = await env.reset(3)
        return sim.state(), obs["obs_str"], (await env.system_prompt())["obs_str"], \
            wm._render(init=True)["obs_str"], (await wm.system_prompt())["obs_str"]

    st, obs, system, wm_obs, wm_system = asyncio.run(run())
    m = _OFFSET_RE.search(obs)
    got = [float(v) for v in (m.group(1) + "," + m.group(2)).split(",")]
    want = [t - c for c, t in zip(st["gripper_pose"], st["target_pose"])]
    assert np.allclose(got, want, atol=0.051) and m.group(3) == "open"
    # Six numbers (xyz + roll/pitch/yaw), never a quaternion.
    assert len(got) == 6 and "quaternion" not in (obs + system).lower()
    assert "<image>" not in obs and "Expert hint" not in obs
    assert len(obs) < 0.3 * len(wm_obs) and len(system) < len(wm_system)


def test_compact_episode_reaches_success_by_cancelling_the_offset():
    from vagen_src.src_env import SRCEnv

    def policy(obs_str):
        m = _OFFSET_RE.search(obs_str)
        return f"<answer>move({m.group(1)})|rotate({m.group(2)})|gripper(close)</answer>"

    async def run():
        env = SRCEnv(COMPACT, sim=_make_sim())
        obs, info = await env.reset(5)
        done, turns = False, 0
        while not done and turns < 8:
            obs, _, done, info = await env.step(policy(obs["obs_str"]))
            turns += 1
            assert "Last action: move(" in obs["obs_str"]
        return info

    assert asyncio.run(run())["success"]


def test_expert_index_follows_the_demonstration_to_the_goal():
    from vagen_src.expert_rag import ExpertIndex

    index = ExpertIndex(_fake_expert(), lookahead=10)
    assert index.n_episodes == 3 and len(index) == 3 * 40
    # On the expert's own start state the hint is the expert's motion over `lookahead` steps.
    start = index.offsets[0]
    hit = index.query(start, 0.8, 20, 30)
    assert hit.match_distance_mm < 1e-6 and [p.name for p in hit.primitives] == ["move", "rotate"]
    assert np.allclose(hit.primitives[0].values, (start - index.offsets[10])[:3], atol=0.051)
    # From a state the expert never visited, repeated hints still end at the goal with the jaw closed.
    offset, jaw = np.array([30.0, -12.0, -35.0, 40.0, 10.0, -25.0]), 0.8
    for _ in range(12):
        for p in index.query(offset, jaw, 20, 30).primitives:
            if p.name == "move":
                offset[:3] -= p.values
            elif p.name == "rotate":
                offset[3:] -= p.values
            else:
                jaw = 0.0
    assert np.linalg.norm(offset[:3]) < 1.0 and np.linalg.norm(offset[3:]) < 2.0 and jaw == 0.0


def test_expert_hint_in_observation_solves_the_episode(tmp_path):
    import pickle

    from vagen_src.src_env import SRCEnv

    path = tmp_path / "expert.pkl"
    path.write_bytes(pickle.dumps(_fake_expert()))
    cfg = {**COMPACT, "rag_hint": True, "rag_lookahead": 15, "rag_expert_path": str(path)}

    async def run():
        env = SRCEnv(cfg, sim=_make_sim())
        assert "Expert hint" in (await env.system_prompt())["obs_str"]
        obs, info = await env.reset(9)
        done, turns = False, 0
        while not done and turns < 8:
            hint = re.search(r"Expert hint: (.*)", obs["obs_str"]).group(1)
            obs, _, done, info = await env.step(f"<answer>{hint}</answer>")
            turns += 1
        return info

    assert asyncio.run(run())["success"]
    with pytest.raises(ValueError):
        SRCEnv({**cfg, "prompt_format": "wm"}, sim=_make_sim())


def test_expert_start_removes_needle_noise_and_checks_the_start(tmp_path):
    import pickle

    from vagen_src.src_env import SRCEnv

    sim = _make_sim()
    nominal = sim.reset(1, expert_start=True)
    assert sim.reset(2, expert_start=True)["target_pose"] == nominal["target_pose"]
    assert sim.reset(1)["target_pose"] != nominal["target_pose"]          # noise is back
    assert sim.reset(1)["target_pose"] == sim.reset(1)["target_pose"]

    def expert(goal_shift_cm):
        cur = np.r_[np.array(nominal["gripper_pose"][:3]) / 10, np.radians(nominal["gripper_pose"][3:]), 0.8]
        goal = np.r_[np.array(nominal["target_pose"][:3]) / 10 + goal_shift_cm, np.radians(nominal["target_pose"][3:]), 0.0]
        step = {"obs": {"achieved_goal": cur, "desired_goal": goal}, "next_obs": {"achieved_goal": goal, "desired_goal": goal},
                "action": np.zeros(7), "reward": np.zeros(1), "done": np.ones(1), "info": {"is_success": True}}
        path = tmp_path / f"expert_{goal_shift_cm}.pkl"
        path.write_bytes(pickle.dumps([step]))
        return str(path)

    async def reset(path, **extra):
        env = SRCEnv({**COMPACT, "expert_start": True, "rag_expert_path": path, **extra}, sim=sim)
        return await env.reset(77)

    _, info = asyncio.run(reset(expert(0.0), expert_start_check_target=True))
    assert info["expert_start_error"]["target_pose_mm"] < 1e-3
    # Demonstrations recorded for another needle pose: reported, and an error when the check is on.
    _, info = asyncio.run(reset(expert(0.5)))
    assert info["expert_start_error"]["target_pose_mm"] == pytest.approx(5.0, abs=1e-3)
    with pytest.raises(RuntimeError):
        asyncio.run(reset(expert(0.5), expert_start_check_target=True))
