#!/usr/bin/env python3
"""
Evaluate SurgicAI's pretrained low-level policies as the baseline.

Uses the checkpoints shipped in third_party/SurgicAI/RL/Evaluation_model/<Task>/<Algo>/
and runs them through the same SRCSim used by the VAGEN server, so both methods
share: the SurgicAI env, per-episode seeds (needle pose), success thresholds,
the 300 low-level step budget and the trajectory-length / time-cost metrics.

Run on the simulator machine with AMBF already running (stop vagen_src.serve
first; only one client should drive the simulator):

    python scripts/eval_surgicai_baseline.py --task approach --algorithm TD3_HER_BC \
        --seeds_from configs/eval_src_approach.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vagen_src.paths import REPO_ROOT, get_path, setup_sim_paths  # noqa: E402

SB3_ALGOS = ("BC", "TD3_BC", "TD3_HER_BC")
D3_ALGOS = ("IQL",)


def load_policy(task: str, algorithm: str, model_path: str = ""):
    """Return a function obs_dict -> action in [-1, 1]^7."""
    model_dir = os.path.join(get_path("SURGICAI_RL_DIR"), "Evaluation_model", task.capitalize(), algorithm)
    if algorithm in SB3_ALGOS:
        from RL_algo.BC import BC
        from RL_algo.td3_BC import TD3_BC

        cls = BC if algorithm == "BC" else TD3_BC
        model = cls.load(model_path or os.path.join(model_dir, "final_model.zip"), device="cpu")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if algorithm in D3_ALGOS:
        import Offline_RL_algo.d3rlpy as d3rlpy

        model = d3rlpy.load_learnable(model_path or os.path.join(model_dir, "final_model.d3"), device=None)
        return lambda obs: model.predict(np.array([obs["observation"]]))[0]
    raise ValueError(f"Unsupported algorithm {algorithm}. Choose from {SB3_ALGOS + D3_ALGOS}")


def read_seeds(args) -> list:
    if args.seeds_from:
        import yaml

        with open(args.seeds_from) as f:
            cfg = yaml.safe_load(f)
        spec = cfg["envs"][0]
        return [int(s) for s in spec["seed_list"][: int(spec["n_envs"])]]
    return list(range(args.seed_start, args.seed_start + args.num_episodes))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="approach")
    p.add_argument("--algorithm", default="TD3_HER_BC", help="BC | TD3_BC | TD3_HER_BC | IQL")
    p.add_argument("--model_path", default="", help="override the shipped checkpoint")
    p.add_argument("--trans_error", type=float, default=0.5, help="cm, must match the VAGEN server")
    p.add_argument("--angle_error", type=float, default=30.0, help="degrees, must match the VAGEN server")
    p.add_argument("--max_low_level_steps", type=int, default=300)
    p.add_argument("--seeds_from", default="", help="eval yaml whose envs[0].seed_list is reused")
    p.add_argument("--seed_start", type=int, default=20000)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "baseline"))
    args = p.parse_args()

    setup_sim_paths()
    from vagen_src.sim import SRCSim

    sim = SRCSim(
        task=args.task,
        trans_error=args.trans_error,
        angle_error=args.angle_error,
        max_low_level_steps=args.max_low_level_steps,
        camera_topic=None,
    )
    policy = load_policy(sim.task, args.algorithm, args.model_path)
    seeds = read_seeds(args)

    episodes = []
    for i, seed in enumerate(seeds):
        sim.reset(seed)
        t0 = time.time()
        while not sim.done:
            sim.low_level_step(policy(sim.env.obs))
        st = sim.state()
        episodes.append({
            "seed": seed,
            "success": bool(st["success"]),
            "low_level_steps": int(st["low_level_steps"]),
            "traj_length_mm": float(st["traj_length_mm"]),
            "trans_error_mm": float(st["trans_error_mm"]),
            "angle_error_deg": float(st["angle_error_deg"]),
            "wall_seconds": time.time() - t0,
        })
        n_succ = sum(e["success"] for e in episodes)
        print(f"[{i + 1}/{len(seeds)}] seed={seed} success={st['success']} "
              f"steps={st['low_level_steps']} running_success={n_succ / len(episodes):.2%}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{sim.task}_{args.algorithm}.json")
    with open(out, "w") as f:
        json.dump({"method": f"SurgicAI-{args.algorithm}", "task": sim.task,
                   "trans_error": args.trans_error, "angle_error": args.angle_error,
                   "max_low_level_steps": args.max_low_level_steps, "episodes": episodes}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
