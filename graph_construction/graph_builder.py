"""
Phase 1 — Hybrid Dependency Graph Builder
==========================================
Fuses static RawEdges and dynamic TraceEdges into a single weighted
NetworkX DiGraph.  Edge-weighting strategy:

  - Static-only edge:     weight = BASE_WEIGHT  (structural mandate)
  - Dynamic-validated:    weight = BASE_WEIGHT + DYNAMIC_SCALE * log(1 + freq)
  - Multiple static kinds (call + extend + field) accumulate multiplicatively

The result is exported to:
  * A NetworkX DiGraph  (for Phase 2 centrality computation)
  * A PyTorch-Geometric Data object  (for GNN ingestion)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from rich.console import Console

from .static_analyzer import RawEdge
from .dynamic_tracer import TraceEdge

console = Console()

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

BASE_WEIGHT: float = 1.0          # baseline for any static edge
DYNAMIC_SCALE: float = 5.0        # amplifier for dynamic confirmation
KIND_MULTIPLIERS: dict[str, float] = {
    "call": 1.0,
    "extend": 0.8,
    "implement": 0.7,
    "field": 0.5,
    "import": 0.3,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class HybridGraph:
    """Container for the fused dependency graph in multiple representations."""

    nx_graph: nx.DiGraph
    # Mapping from integer node index → fully-qualified class name
    idx_to_class: dict[int, str]
    # Reverse mapping
    class_to_idx: dict[str, int]
    # PyG Data object (populated lazily via `.to_pyg()`)
    _pyg: Optional[Data] = field(default=None, repr=False)

    @property
    def num_nodes(self) -> int:
        return self.nx_graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.nx_graph.number_of_edges()

    def to_pyg(self) -> Data:
        """Convert to a PyTorch-Geometric Data object (cached)."""
        if self._pyg is not None:
            return self._pyg

        G = self.nx_graph
        node_list = sorted(G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        edge_index_list, edge_weight_list = [], []
        for u, v, d in G.edges(data=True):
            edge_index_list.append([node_map[u], node_map[v]])
            edge_weight_list.append(d.get("weight", 1.0))

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        num_nodes = len(node_list)

        # Degree as initial node feature [in_degree, out_degree]
        in_deg  = torch.tensor([G.in_degree(n)  for n in node_list], dtype=torch.float).unsqueeze(1)
        out_deg = torch.tensor([G.out_degree(n) for n in node_list], dtype=torch.float).unsqueeze(1)
        x = torch.cat([in_deg, out_deg], dim=1)

        self._pyg = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=num_nodes,
        )
        return self._pyg

    def summary(self) -> str:
        G = self.nx_graph
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        return (
            f"HybridGraph | nodes={G.number_of_nodes():,}  "
            f"edges={G.number_of_edges():,}  "
            f"avg_weight={float(np.mean(weights)):.2f}  "
            f"max_weight={float(np.max(weights)):.2f}"
        )


def build_hybrid_graph(
    static_edges: list[RawEdge],
    dynamic_edges: list[TraceEdge],
    base_weight: float = BASE_WEIGHT,
    dynamic_scale: float = DYNAMIC_SCALE,
    filter_external_packages: Optional[list[str]] = None,
) -> HybridGraph:
    """
    Fuse static and dynamic edges into a single weighted DiGraph.

    Parameters
    ----------
    static_edges :
        Edges from java-callgraph / Jarviz / ObjectAid.
    dynamic_edges :
        Edges from OTel / Mono2Micro with execution frequencies.
    base_weight :
        Minimum weight assigned to any structurally declared edge.
    dynamic_scale :
        Amplification factor applied to log-normalized runtime frequency.
    filter_external_packages :
        If provided, any class whose FQN starts with one of these prefixes
        (e.g. ["java.", "javax.", "org.springframework."]) is excluded.
        This keeps the graph focused on application code.
    """

    def _is_external(cls: str) -> bool:
        if not filter_external_packages:
            return False
        return any(cls.startswith(pkg) for pkg in filter_external_packages)

    # ------------------------------------------------------------------ #
    # 1. Accumulate static edge weights by (src, tgt) pair               #
    # ------------------------------------------------------------------ #
    static_acc: dict[tuple[str, str], float] = defaultdict(float)
    for e in static_edges:
        if _is_external(e.source) or _is_external(e.target):
            continue
        mult = KIND_MULTIPLIERS.get(e.kind, 1.0)
        static_acc[(e.source, e.target)] += base_weight * mult

    # ------------------------------------------------------------------ #
    # 2. Build dynamic lookup                                             #
    # ------------------------------------------------------------------ #
    dyn_freq: dict[tuple[str, str], int] = defaultdict(int)
    for e in dynamic_edges:
        if _is_external(e.source) or _is_external(e.target):
            continue
        dyn_freq[(e.source, e.target)] += e.count

    # ------------------------------------------------------------------ #
    # 3. Merge into unified edge weight table                             #
    # ------------------------------------------------------------------ #
    all_pairs: set[tuple[str, str]] = set(static_acc.keys()) | set(dyn_freq.keys())
    edge_weights: dict[tuple[str, str], float] = {}

    for pair in all_pairs:
        w = static_acc.get(pair, base_weight * 0.5)  # dynamic-only edge: half base
        freq = dyn_freq.get(pair, 0)
        if freq > 0:
            w += dynamic_scale * math.log1p(freq)
        edge_weights[pair] = w

    # ------------------------------------------------------------------ #
    # 4. Construct DiGraph                                                #
    # ------------------------------------------------------------------ #
    G = nx.DiGraph()
    for (src, tgt), w in edge_weights.items():
        G.add_edge(src, tgt, weight=w)

    # Build index maps
    sorted_nodes = sorted(G.nodes())
    idx_to_class = {i: n for i, n in enumerate(sorted_nodes)}
    class_to_idx = {n: i for i, n in idx_to_class.items()}

    stats = {
        "static_edges": len(static_acc),
        "dynamic_edges": len(dyn_freq),
        "dynamic_confirmed_static": sum(1 for p in static_acc if p in dyn_freq),
        "dynamic_only": sum(1 for p in dyn_freq if p not in static_acc),
    }

    console.print(
        f"[bold cyan]HybridGraph built:[/]\n"
        f"  nodes                 = {G.number_of_nodes():,}\n"
        f"  edges                 = {G.number_of_edges():,}\n"
        f"  static edges          = {stats['static_edges']:,}\n"
        f"  dynamic edges         = {stats['dynamic_edges']:,}\n"
        f"  dynamic-confirmed     = {stats['dynamic_confirmed_static']:,}\n"
        f"  dynamic-only          = {stats['dynamic_only']:,}"
    )

    return HybridGraph(nx_graph=G, idx_to_class=idx_to_class, class_to_idx=class_to_idx)


def save_graph(hg: HybridGraph, path: str) -> None:
    """Serialize the DiGraph to GraphML format."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nx.write_graphml(hg.nx_graph, path)
    console.print(f"[green]Graph saved to[/] {path}")


def load_graph(path: str) -> HybridGraph:
    """Deserialize a GraphML file back into a HybridGraph."""
    G = nx.read_graphml(path)
    sorted_nodes = sorted(G.nodes())
    idx_to_class = {i: n for i, n in enumerate(sorted_nodes)}
    class_to_idx = {n: i for i, n in idx_to_class.items()}
    console.print(f"[green]Graph loaded:[/] {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return HybridGraph(nx_graph=G, idx_to_class=idx_to_class, class_to_idx=class_to_idx)
