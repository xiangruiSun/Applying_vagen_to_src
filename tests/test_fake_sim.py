"""
Offline test of the glue code with a fake SurgicAI env (no ROS / AMBF needed).

Checks primitive parsing, primitive -> low-level step expansion, SRCEnv's
VAGEN interface and the full HTTP path through VAGEN's GymService and
GymImageEnvClient. Run with the simulator-side Python (3.8):

    pip install pytest numpy pillow fastapi "uvicorn<0.41" httpx
    python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
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

    def reset(self, **kwargs):
        start = np.array([0.0, 0.0, -0.08, -3.0, 0.0, 1.0, 0.8], dtype=np.float32)
        noise = np.r_[np.random.uniform(-0.02, 0.02, 3), np.random.uniform(-0.3, 0.3, 3), 0.0]
        self.psm_goal_list[1] = start
        self.goal_obs = (start + noise).astype(np.float32)
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
    r = parse_response("<observation>a</observation><think>b</think><answer>move(1,2,3)|gripper(close)</answer>"
                       "<prediction>c</prediction>", "wm")
    assert r["format_correct"] and r["actions"] == ["move(1,2,3)", "gripper(close)"]
    assert parse_primitive("rotate(5, -3, 0)").values == (5.0, -3.0, 0.0)
    assert parse_primitive("gripper(open)").jaw_open is True
    assert parse_primitive("move(1,2)") is None
    assert parse_primitive("fly(1,2,3)") is None
    assert not parse_response("move(1,2,3)", "wm")["format_correct"]


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

    info, total = asyncio.get_event_loop().run_until_complete(run())
    assert info["success"] and info["metrics"]["traj_metrics"]["success"]
    assert 0 < info["low_level_steps"] <= 300
    assert total > 10


def test_invalid_answer_gets_no_format_reward():
    from vagen_src.src_env import SRCEnv

    async def run():
        env = SRCEnv(CFG, sim=_make_sim())
        await env.reset(1)
        return await env.step("I would move left")

    _, reward, done, info = asyncio.get_event_loop().run_until_complete(run())
    assert reward == 0.0 and not done and info["low_level_steps"] == 0


def test_remote_roundtrip_through_vagen_service():
    import uvicorn
    from vagen.envs_remote import GymImageEnvClient, GymService

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
        results = asyncio.get_event_loop().run_until_complete(run())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert all(r["success"] for r in results)
