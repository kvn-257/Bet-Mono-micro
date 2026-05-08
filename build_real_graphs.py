"""
Build Real-World Graphs
=======================
Scans all 4 Java repos in data/ using build_graph_from_repo(),
then saves each resulting DiGraph as a NetworkX gpickle file so
the GNN training loop can load and evaluate on them without
re-parsing the Java sources every run.

Output: real_world_graphs/<repo_name>.gpickle

Usage:
    python build_real_graphs.py
"""

import sys
import pickle
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_construction.graph_analysis import build_graph_from_repo

REPOS = [
    "acmeair",
    "jpetstore-6",
    "sample.daytrader7",
    "sample.plantsbywebsphere",
]

DATA_DIR  = PROJECT_ROOT / "data"
OUT_DIR   = PROJECT_ROOT / "real_world_graphs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  Building real-world dependency graphs")
    print("=" * 60)

    for repo_name in REPOS:
        repo_path = DATA_DIR / repo_name
        if not repo_path.exists():
            print(f"[SKIP] {repo_name}: directory not found at {repo_path}")
            continue

        print(f"\n[{repo_name}]")
        try:
            G = build_graph_from_repo(repo_path, filter_external=False)
            print(f"  -> {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Save
            out_path = OUT_DIR / f"{repo_name}.gpickle"
            with open(out_path, "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  -> Saved to {out_path}")

        except Exception as e:
            print(f"  [ERROR] Failed to build graph for {repo_name}: {e}")

    print("\n[DONE] All real-software graphs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
