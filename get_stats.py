import sys
import json
from pathlib import Path
sys.path.append('E:\\Bet-Mono-micro')
from graph_construction.graph_analysis import build_graph_from_repo, topology_report, detect_communities

data_dir = Path('E:\\Bet-Mono-micro\\data')
repos = ['acmeair', 'jpetstore-6', 'sample.daytrader7', 'sample.plantsbywebsphere']

results = {}
for repo in repos:
    G = build_graph_from_repo(data_dir / repo, filter_external=False)
    report = topology_report(G)
    comms = detect_communities(G)
    report['num_communities'] = len(set(comms.values()))
    results[repo] = report

with open('E:\\Bet-Mono-micro\\stats_out.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Done!")
