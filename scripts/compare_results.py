#!/usr/bin/env python3
"""
Compare VAGEN rollouts against SurgicAI baselines with SurgicAI's metrics:
  - success rate
  - average trajectory length (mm, successful episodes only)
  - average time cost (low-level steps, successful episodes only)
plus paired success on the seeds both methods evaluated.

    python scripts/compare_results.py \
        --vagen results/vagen_eval/src_approach \
        --baseline results/baseline/approach_TD3_HER_BC.json results/baseline/approach_BC.json \
        --out results/compare_approach.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_vagen(dump_dir: str, name: str) -> Dict:
    episodes = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "**", "metrics.json"), recursive=True)):
        with open(path) as f:
            m = json.load(f)
        # Use the env's own success flag (VAGEN's eval may infer success from reward).
        infos = [i for i in m.get("infos", []) if isinstance(i, dict) and "low_level_steps" in i]
        if not infos:
            continue
        last = infos[-1]
        episodes.append({
            "seed": m.get("seed"),
            "success": bool(last.get("success", False)),
            "low_level_steps": int(last["low_level_steps"]),
            "traj_length_mm": float(last["traj_length_mm"]),
            "num_turns": int(m.get("num_turns", 0)),
        })
    if not episodes:
        raise SystemExit(f"No metrics.json with SRC infos found under {dump_dir}")
    return {"method": name, "episodes": episodes}


def summarize(run: Dict) -> Dict:
    eps: List[Dict] = run["episodes"]
    succ = [e for e in eps if e["success"]]
    row = {
        "method": run["method"],
        "episodes": len(eps),
        "success_rate": len(succ) / len(eps),
        "traj_length_mm": float(np.mean([e["traj_length_mm"] for e in succ])) if succ else float("nan"),
        "traj_length_std": float(np.std([e["traj_length_mm"] for e in succ])) if succ else float("nan"),
        "time_cost_steps": float(np.mean([e["low_level_steps"] for e in succ])) if succ else float("nan"),
        "time_cost_std": float(np.std([e["low_level_steps"] for e in succ])) if succ else float("nan"),
    }
    if eps and "num_turns" in eps[0]:
        row["avg_vlm_turns"] = float(np.mean([e["num_turns"] for e in eps]))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vagen", nargs="*", default=[], help="VAGEN eval dump dir(s); name with dir=label")
    p.add_argument("--baseline", nargs="*", default=[], help="JSON files from eval_surgicai_baseline.py")
    p.add_argument("--out", default="", help="optional CSV path")
    args = p.parse_args()

    runs = []
    for spec in args.vagen:
        path, _, label = spec.partition("=")
        runs.append(load_vagen(path, label or f"VAGEN ({os.path.basename(path.rstrip('/'))})"))
    for path in args.baseline:
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit("Nothing to compare: pass --vagen and/or --baseline")

    rows = [summarize(r) for r in runs]
    print("| Method | Episodes | Success | Traj. length (mm) | Time cost (steps) | VLM turns |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        turns = f"{r['avg_vlm_turns']:.1f}" if "avg_vlm_turns" in r else "-"
        print(f"| {r['method']} | {r['episodes']} | {r['success_rate']:.1%} | "
              f"{r['traj_length_mm']:.1f} ± {r['traj_length_std']:.1f} | "
              f"{r['time_cost_steps']:.1f} ± {r['time_cost_std']:.1f} | {turns} |")

    # Paired comparison on shared seeds.
    by_seed = [{e["seed"]: e["success"] for e in r["episodes"]} for r in runs]
    shared = set.intersection(*(set(d) for d in by_seed)) if len(runs) > 1 else set()
    if shared:
        print(f"\nPaired success on {len(shared)} shared seeds:")
        for r, d in zip(runs, by_seed):
            print(f"  {r['method']}: {np.mean([d[s] for s in shared]):.1%}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
