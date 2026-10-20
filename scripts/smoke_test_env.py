#!/usr/bin/env python3
"""
End-to-end check of the SRC <-> VAGEN glue without any VLM.

Mode `local`  (simulator machine, AMBF running, server NOT running):
    builds SRCEnv in-process and drives it with a scripted "oracle" that
    writes VLM-formatted answers moving the gripper toward the target pose.

Mode `remote` (any machine with VAGEN installed, server running):
    talks to `python -m vagen_src.serve` through VAGEN's own RemoteEnv
    client, i.e. exactly the path used by training and evaluation.

    python scripts/smoke_test_env.py --mode local --task approach --episodes 2
    python scripts/smoke_test_env.py --mode remote --base_url http://<sim-host>:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vagen_src.paths import REPO_ROOT  # noqa: E402

_POSE_RE = re.compile(r"{}:\s+position \(([^)]*)\) mm, orientation \(([^)]*)\) deg")


def oracle_answer(obs_str: str, max_move: float, max_rot: float) -> str:
    """Move toward the target shown in the observation text, then close the jaw."""
    def pose(label):
        m = re.search(_POSE_RE.pattern.format(label), obs_str)
        return [float(v) for v in (m.group(1) + "," + m.group(2)).split(",")] if m else None

    cur, tgt = pose("Gripper pose"), pose("Target pose")
    if cur is None or tgt is None:
        return "<observation>?</observation><think>no target</think><answer>gripper(close)</answer><prediction>?</prediction>"
    d = [t - c for c, t in zip(cur, tgt)]
    clip = lambda v, m: max(-m, min(m, v))
    acts = []
    if max(abs(v) for v in d[:3]) > 0.5:
        acts.append("move({:.1f}, {:.1f}, {:.1f})".format(*(clip(v, max_move) for v in d[:3])))
    if max(abs(v) for v in d[3:6]) > 2.0:
        acts.append("rotate({:.1f}, {:.1f}, {:.1f})".format(*(clip(v, max_rot) for v in d[3:6])))
    acts.append("gripper(close)" if not acts else "gripper(open)")
    return ("<observation>The gripper is above the phantom.</observation>"
            "<think>Move toward the target pose.</think>"
            f"<answer>{'|'.join(acts)}</answer>"
            "<prediction>The gripper will be closer to the target.</prediction>")


async def run_episodes(env, episodes: int, max_turns: int, seed0: int, save_dir: str) -> None:
    print((await env.system_prompt())["obs_str"][:600], "...\n")
    for ep in range(episodes):
        obs, info = await env.reset(seed0 + ep)
        for turn in range(max_turns):
            _save(obs, save_dir, ep, turn)
            print(f"--- ep {ep} turn {turn} ---\n{obs['obs_str'][:400]}")
            answer = oracle_answer(obs["obs_str"], 20, 30)
            obs, reward, done, info = await env.step(answer)
            print(f">>> {answer}\n    reward={reward:.3f} done={done} success={info.get('success')} "
                  f"steps={info.get('low_level_steps')} err={info.get('trans_error_mm', 0):.1f}mm")
            if done:
                break
    await env.close()


def _save(obs, save_dir, ep, turn) -> None:
    imgs = obs.get("multi_modal_input", {}).get("<image>", [])
    if imgs and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        imgs[0].save(os.path.join(save_dir, f"ep{ep}_turn{turn}.png"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["local", "remote"], default="local")
    p.add_argument("--task", default="approach")
    p.add_argument("--render_mode", default="vision", choices=["vision", "text"])
    p.add_argument("--base_url", default="http://localhost:8000")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max_turns", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=os.path.join(REPO_ROOT, "results", "smoke_test"))
    args = p.parse_args()

    cfg = {"task": args.task, "render_mode": args.render_mode, "prompt_format": "wm"}
    if args.mode == "local":
        from vagen_src.paths import setup_sim_paths

        setup_sim_paths()
        from vagen_src.sim import SRCSim
        from vagen_src.src_env import SRCEnv

        camera = "/ambf/env/cameras/cameraL/ImageData" if args.render_mode == "vision" else None
        env = SRCEnv(cfg, sim=SRCSim(task=args.task, camera_topic=camera))
    else:
        from vagen_src.paths import setup_vagen_path

        setup_vagen_path()
        from vagen.envs_remote import GymImageEnvClient

        env = GymImageEnvClient({"base_urls": args.base_url, "timeout": 600, "retries": 3, **cfg})

    asyncio.get_event_loop().run_until_complete(
        run_episodes(env, args.episodes, args.max_turns, args.seed, args.save_dir)
    )


if __name__ == "__main__":
    main()
