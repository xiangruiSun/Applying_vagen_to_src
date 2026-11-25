#!/usr/bin/env python3
"""
Success rate of scripted policies on a running SRC env server (no language model).

  rag     answer with the server's "Expert hint" (expert retrieval, vagen_src/expert_rag.py)
  offset  cancel the offset shown in the compact observation (clipped by the server)

Both read the compact observation, so they are the ceilings a model trained on that
prompt can copy. Same client path as training (VAGEN's RemoteEnv).

    python scripts/eval_scripted.py --policy rag --seeds 10000 10015 --rag_lookahead 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vagen_src.paths import setup_vagen_path  # noqa: E402

_OFFSET_RE = re.compile(r"Offset: position \(([^)]*)\) mm, orientation \(([^)]*)\) deg")
_HINT_RE = re.compile(r"Expert hint: (.*)")


def answer(policy: str, obs_str: str) -> str:
    if policy == "rag":
        return f"<answer>{_HINT_RE.search(obs_str).group(1)}</answer>"
    m = _OFFSET_RE.search(obs_str)
    return f"<answer>move({m.group(1)})|rotate({m.group(2)})|gripper(close)</answer>"


async def run(args) -> list:
    from vagen.envs.remote import GymImageEnvClient

    cfg = {"base_urls": args.base_url, "timeout": 600, "retries": 3, "task": args.task,
           "render_mode": "text", "prompt_format": "compact",
           "rag_hint": args.policy == "rag", "rag_lookahead": args.rag_lookahead,
           "expert_start": args.expert_start}
    rows = []
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        env = GymImageEnvClient(cfg)
        obs, info = await env.reset(seed)
        first, total, turns, done = obs["obs_str"], 0.0, 0, False
        while not done and turns < args.max_turns:
            obs, reward, done, info = await env.step(answer(args.policy, obs["obs_str"]))
            total += reward
            turns += 1
        await env.close()
        rows.append({"seed": seed, "success": bool(info.get("success")), "turns": turns,
                     "low_level_steps": info.get("low_level_steps"), "reward": round(total, 3),
                     "trans_error_mm": round(info.get("trans_error_mm", 0.0), 2),
                     "angle_error_deg": round(info.get("angle_error_deg", 0.0), 2), "first_obs": first})
        print({k: v for k, v in rows[-1].items() if k != "first_obs"}, flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["rag", "offset"], default="rag")
    p.add_argument("--task", default="approach")
    p.add_argument("--base_url", default="http://localhost:8000")
    p.add_argument("--seeds", type=int, nargs=2, default=[10000, 10015], metavar=("FIRST", "LAST"))
    p.add_argument("--max_turns", type=int, default=8)
    p.add_argument("--rag_lookahead", type=int, default=15)
    p.add_argument("--expert_start", action="store_true", help="start every episode at the expert's start")
    p.add_argument("--out", default="")
    args = p.parse_args()
    setup_vagen_path()
    rows = asyncio.run(run(args))
    n, ok = len(rows), sum(r["success"] for r in rows)
    summary = {"policy": args.policy, "rag_lookahead": args.rag_lookahead if args.policy == "rag" else None,
               "expert_start": args.expert_start,
               "episodes": n, "success": ok, "success_rate": ok / n,
               "mean_turns": sum(r["turns"] for r in rows) / n,
               "mean_low_level_steps": sum(r["low_level_steps"] for r in rows) / n}
    print(json.dumps(summary))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "episodes": rows}, f, indent=1)


if __name__ == "__main__":
    main()
