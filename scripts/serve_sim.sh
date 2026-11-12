#!/usr/bin/env bash
# Start the VAGEN env server next to a running simulator (scripts/launch_sim.sh).
#   bash scripts/serve_sim.sh --task approach --port 8000
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/sim_env.sh"
cd "$REPO_ROOT"
mkdir -p results
"$SIM_PYTHON" -m vagen_src.serve "$@" 2>&1 | tee "results/serve_$(date +%Y%m%d_%H%M%S).log"
