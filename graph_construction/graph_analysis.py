# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

# Fix stdout encoding on Windows (cp1252) so UTF-8 chars print cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

"""
Graph Analysis -- Dependency Graph Construction, Centrality & Visualization
============================================================================
Builds a class-level dependency graph from a Java repository by statically
scanning .java source files, then performs a full suite of graph-theoretic
analyses (centrality, community, density, path metrics) and renders
publication-quality visualisations.

Usage
-----
    python -m graph_construction.graph_analysis            # uses default repo
    python -m graph_construction.graph_analysis --repo acmeair
    python -m graph_construction.graph_analysis --repo jpetstore-6
    python -m graph_construction.graph_analysis --repo sample.daytrader7
    python -m graph_construction.graph_analysis --repo sample.plantsbywebsphere

All plots are saved to  output/graph_analysis/<repo>/ and also displayed.
"""

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ROOT = Path(__file__).resolve().parent.parent          # â€¦/New folder
_DATA = _ROOT / "data"
_OUT  = _ROOT / "output" / "graph_analysis"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Java source parser â€” produces RawEdge-compatible tuples
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Regex helpers
_PKG_RE         = re.compile(r"^\s*package\s+([\w.]+)\s*;")
# FIX: Capture both regular imports AND wildcard imports separately
_IMPORT_RE      = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)(?:\.\*)?;")
_WILDCARD_IMP_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\.\*;", re.MULTILINE)
_CLASS_RE   = re.compile(
    r"(?:^|\s)(?:public|private|protected|abstract|final|static)?\s*"
    r"(?:class|interface|enum|record)\s+(\w+)"
    r"(?:\s+extends\s+([\w,\s]+?))?(?:\s+implements\s+([\w,\s,<>]+?))?\s*[{<]",
    re.MULTILINE,
)
# FIX 1: Catch ALL field declarations - 'private static final', package-private,
# annotated fields (@Inject/@Autowired). Old regex missed ~60% of fields.
_FIELD_RE   = re.compile(
    r"(?:(?:private|protected|public|static|final|transient|volatile)\s+)*"
    r"([A-Z][\w<>\[\]]*)\s+\w+\s*[;=,]"
)
# FIX 3: Method parameter types for 'uses' edges
_METHOD_PARAM_RE = re.compile(
    r"(?:void|[A-Z][\w<>\[\]]*?)\s+\w+\s*\(([^)]*)\)",
    re.MULTILINE,
)
_ANNO_RE    = re.compile(r"@\w+")
# NEW: Capture `new SomeClass(` expressions anywhere in source (method bodies)
_NEW_INST_RE = re.compile(r"\bnew\s+([A-Z]\w+)\s*[\(<]")
# NEW: Capture static calls `SomeClass.method(` (class is capital letter start)
_STATIC_CALL_RE = re.compile(r"\b([A-Z]\w+)\.(?!class\b)\w+\s*\(")

JAVA_PRIMITIVES = {
    "int","long","double","float","boolean","char","byte","short","void",
    "String","Object","Integer","Long","Double","Float","Boolean","List",
    "Map","Set","Collection","Optional","Stream","Iterator","Iterable",
    "Comparable","Serializable","Exception","RuntimeException","Throwable",
    "Class","Enum","Number","Byte","Character","Short","Void",
    "StringBuilder","StringBuffer","Thread","Runnable",
}


def _fqn(pkg: str, simple: str) -> str:
    return f"{pkg}.{simple}" if pkg else simple


def _strip_generics(name: str) -> str:
    return re.sub(r"<.*?>", "", name).strip()


def scan_java_file(path: Path) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Parse one .java file.
    Returns fqn (fully-qualified class name) and list of (src, tgt, kind) edges.
    kind in: import, extend, implement, field, uses
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", []

    pkg = ""
    m = _PKG_RE.search(raw)
    if m:
        pkg = m.group(1)

    imports: dict[str, str] = {}
    for m in _IMPORT_RE.finditer(raw):
        fq = m.group(1)
        simple = fq.rsplit(".", 1)[-1]
        imports[simple] = fq

    primary_simple = path.stem
    primary_fqn = _fqn(pkg, primary_simple)
    edges: list[tuple[str, str, str]] = []

    # FIX 2: _resolve maps simple names -> FQN via imports OR same-package assumption.
    # Critical for multi-module repos where siblings share a package but have no imports.
    def _resolve(simple: str):
        simple = _strip_generics(simple).strip()
        if not simple or simple in JAVA_PRIMITIVES:
            return None
        if not simple[0].isupper():
            return None
        if simple in imports:
            return imports[simple]
        # Same-package fallback: resolves intra-module sibling references
        if pkg:
            return f"{pkg}.{simple}"
        return simple

    # class-level: extends / implements
    for m in _CLASS_RE.finditer(raw):
        cls_name = m.group(1)
        extends_raw    = m.group(2) or ""
        implements_raw = m.group(3) or ""
        cls_fqn = _fqn(pkg, cls_name)
        for parent in re.split(r"[\s,]+", extends_raw):
            tgt = _resolve(parent)
            if tgt and tgt != cls_fqn:
                edges.append((cls_fqn, tgt, "extend"))
        for iface in re.split(r"[\s,<>]+", implements_raw):
            tgt = _resolve(iface)
            if tgt and tgt != cls_fqn:
                edges.append((cls_fqn, tgt, "implement"))

    # FIX 2: import edges - skip JDK/test, keep all app/framework imports.
    # Every non-JDK import is an explicit compile-time dependency.
    _SKIP_PREFIXES = ("java.", "javax.", "jakarta.", "sun.", "com.sun.",
                      "org.w3c.", "org.xml.", "junit.", "org.junit.")
    for simple, fq in imports.items():
        if fq != primary_fqn and not any(fq.startswith(p) for p in _SKIP_PREFIXES):
            edges.append((primary_fqn, fq, "import"))

    # FIX 1: field-type edges - catches ALL field declarations
    # including private static final, package-private, @Inject annotated fields.
    # Tokenizes generics: handles Map<K,V>, List<Foo> etc.
    for m in _FIELD_RE.finditer(raw):
        raw_type = m.group(1).strip()
        for token in re.split(r"[<>,\s]+", raw_type):
            tgt = _resolve(token)
            if tgt and tgt != primary_fqn:
                edges.append((primary_fqn, tgt, "field"))

    # FIX 3: method parameter type edges (uses relationship)
    # Captures types appearing in method signatures.
    for m in _METHOD_PARAM_RE.finditer(raw):
        param_str = m.group(1)
        for token in re.split(r"[,<>\[\]\s]+", param_str):
            tgt = _resolve(token)
            if tgt and tgt != primary_fqn:
                edges.append((primary_fqn, tgt, "uses"))

    # NEW: new_instance edges — `new SomeClass(...)` anywhere in method bodies
    # Critical for M2MS: direct instantiation = hardest coupling to break across a boundary.
    for m in _NEW_INST_RE.finditer(raw):
        tgt = _resolve(m.group(1))
        if tgt and tgt != primary_fqn:
            edges.append((primary_fqn, tgt, "new_instance"))

    # NEW: static_call edges — `SomeClass.method(` references
    # Captures utility/factory calls not visible through field or import edges alone.
    for m in _STATIC_CALL_RE.finditer(raw):
        tgt = _resolve(m.group(1))
        if tgt and tgt != primary_fqn:
            edges.append((primary_fqn, tgt, "static_call"))

    return primary_fqn, edges

def build_graph_from_repo(repo_path: Path,
                          filter_external: bool = False) -> nx.DiGraph:
    """
    Walk every .java file in `repo_path`, parse edges, and return a DiGraph.

    Edge weights follow the same strategy as graph_builder.py:
        static-only: BASE=1.0 * KIND_MULT
        no dynamic data available here, so only static weights are used.
    """
    KIND_W = {
        "extend":       2.0,   # Inheritance = total structural coupling (hardest to split)
        "implement":    1.5,   # Interface contract coupling
        "field":        1.2,   # Class-scope dependency, persistent coupling
        "new_instance": 1.0,   # Direct instantiation = hard coupling
        "import":       0.6,   # Explicit compile-time dependency
        "uses":         0.4,   # Method-level type reference
        "static_call":  0.3,   # Usually utility/helper, weaker coupling
    }
    acc: dict[tuple[str, str], float] = defaultdict(float)
    kind_map: dict[tuple[str, str], str] = {}

    java_files = list(repo_path.rglob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No .java files found under {repo_path}")

    print(f"  Scanning {len(java_files):,} .java files â€¦")

    all_fqns: set[str] = set()
    raw_edges: list[tuple[str, str, str]] = []

    # First pass: collect all FQNs (needed for wildcard import resolution)
    wildcard_imports: list[tuple[str, str]] = []   # (source_fqn, target_package)
    for jf in java_files:
        fqn, edges = scan_java_file(jf)
        if fqn:
            all_fqns.add(fqn)
        raw_edges.extend(edges)

        # Collect wildcard imports for post-processing
        try:
            raw_text = jf.read_text(encoding="utf-8", errors="replace")
            for wm in _WILDCARD_IMP_RE.finditer(raw_text):
                wildcard_imports.append((fqn, wm.group(1)))
        except Exception:
            pass

    # Second pass: resolve wildcard imports to actual classes in that package
    # e.g. `import com.acmeair.service.*` -> edge to every class in com.acmeair.service
    pkg_to_fqns: dict[str, list[str]] = {}
    for fqn in all_fqns:
        pkg = fqn.rsplit(".", 1)[0] if "." in fqn else ""
        pkg_to_fqns.setdefault(pkg, []).append(fqn)

    for source_fqn, target_pkg in wildcard_imports:
        if target_pkg in pkg_to_fqns:
            for target_fqn in pkg_to_fqns[target_pkg]:
                if target_fqn != source_fqn:
                    raw_edges.append((source_fqn, target_fqn, "import"))

    if filter_external and all_fqns:
        raw_edges = [(s, t, k) for s, t, k in raw_edges
                     if s in all_fqns and t in all_fqns]

    for src, tgt, kind in raw_edges:
        acc[(src, tgt)] += KIND_W.get(kind, 1.0)
        kind_map.setdefault((src, tgt), kind)

    G = nx.DiGraph()
    for (src, tgt), w in acc.items():
        G.add_edge(src, tgt, weight=w, kind=kind_map.get((src, tgt), "import"))

    # ensure isolated nodes (files with no edges) are also in the graph
    for fqn in all_fqns:
        G.add_node(fqn)

    return G


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Centrality & topology metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_centralities(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute an exhaustive set of centrality measures and return as a DataFrame.
    """
    print("  Computing centralities â€¦")
    U = G.to_undirected()

    metrics: dict[str, dict] = {}

    # 1. Degree centralities
    metrics["degree_centrality"]     = nx.degree_centrality(G)
    metrics["in_degree_centrality"]  = nx.in_degree_centrality(G)
    metrics["out_degree_centrality"] = nx.out_degree_centrality(G)
    metrics["in_degree"]   = dict(G.in_degree())
    metrics["out_degree"]  = dict(G.out_degree())
    metrics["total_degree"] = {n: G.in_degree(n) + G.out_degree(n)
                                for n in G.nodes()}

    # 2. Betweenness centrality (with/without weight)
    print("    betweenness â€¦")
    metrics["betweenness_centrality"] = nx.betweenness_centrality(
        G, weight=None, normalized=True)
    metrics["betweenness_weighted"] = nx.betweenness_centrality(
        G, weight="weight", normalized=True)

    # 3. Closeness centrality
    print("    closeness â€¦")
    metrics["closeness_centrality"] = nx.closeness_centrality(G)

    # 4. Eigenvector centrality (undirected version for stability)
    print("    eigenvector â€¦")
    try:
        metrics["eigenvector_centrality"] = nx.eigenvector_centrality(
            U, max_iter=500, tol=1e-4, weight="weight")
    except nx.PowerIterationFailedConvergence:
        metrics["eigenvector_centrality"] = dict.fromkeys(G.nodes(), 0.0)

    # 5. PageRank
    print("    pagerank â€¦")
    metrics["pagerank"] = nx.pagerank(G, alpha=0.85, weight="weight",
                                       max_iter=200)

    # 6. HITS (hubs & authorities)
    print("    HITS â€¦")
    try:
        hubs, auth = nx.hits(G, max_iter=200, normalized=True)
        metrics["hub_score"]       = hubs
        metrics["authority_score"] = auth
    except nx.PowerIterationFailedConvergence:
        metrics["hub_score"]       = dict.fromkeys(G.nodes(), 0.0)
        metrics["authority_score"] = dict.fromkeys(G.nodes(), 0.0)

    # 7. Katz centrality
    print("    Katz â€¦")
    try:
        metrics["katz_centrality"] = nx.katz_centrality(
            G, alpha=0.01, beta=1.0, max_iter=1000, normalized=True)
    except nx.PowerIterationFailedConvergence:
        metrics["katz_centrality"] = dict.fromkeys(G.nodes(), 0.0)

    # 8. Load centrality
    print("    load centrality â€¦")
    metrics["load_centrality"] = nx.load_centrality(G)

    # 9. Harmonic centrality
    print("    harmonic â€¦")
    metrics["harmonic_centrality"] = nx.harmonic_centrality(G)

    # 10. Clustering coefficient (undirected)
    print("    clustering â€¦")
    metrics["clustering"] = nx.clustering(U, weight="weight")

    # 11. Triangles
    metrics["triangles"] = nx.triangles(U)

    # 12. Constraint / structural holes (Burt's constraint)
    print("    Burt constraint â€¦")
    try:
        metrics["burt_constraint"] = nx.constraint(U, weight="weight")
    except Exception:
        metrics["burt_constraint"] = dict.fromkeys(G.nodes(), float("nan"))

    # 13. Squad metric: weighted sum of edges (fan-out)
    metrics["weighted_out"] = {
        n: sum(d.get("weight", 1.0) for _, _, d in G.out_edges(n, data=True))
        for n in G.nodes()
    }
    metrics["weighted_in"] = {
        n: sum(d.get("weight", 1.0) for _, _, d in G.in_edges(n, data=True))
        for n in G.nodes()
    }

    # Build DataFrame
    df = pd.DataFrame(metrics)
    df.index.name = "class"
    df = df.fillna(0.0)
    return df


def topology_report(G: nx.DiGraph) -> dict[str, Any]:
    """Top-level structural statistics."""
    U   = G.to_undirected()
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))

    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]

    report: dict[str, Any] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_weight": float(np.mean(weights)) if weights else 0.0,
        "max_weight": float(np.max(weights))  if weights else 0.0,
        "weakly_connected_components":  len(wcc),
        "strongly_connected_components": len(scc),
        "largest_wcc_size": max((len(c) for c in wcc), default=0),
        "largest_scc_size": max((len(c) for c in scc), default=0),
        "is_dag": nx.is_directed_acyclic_graph(G),
        "self_loops": nx.number_of_selfloops(G),
    }

    # Diameter / avg shortest path only on largest WCC subgraph (undirected)
    if U.number_of_nodes() > 1:
        giant = U.subgraph(max(nx.connected_components(U), key=len))
        try:
            report["diameter"] = nx.diameter(giant)
            report["avg_shortest_path"] = nx.average_shortest_path_length(giant)
        except Exception:
            report["diameter"] = "N/A"
            report["avg_shortest_path"] = "N/A"

    # Average clustering
    report["avg_clustering"] = nx.average_clustering(U, weight="weight")

    # Reciprocity
    report["reciprocity"] = nx.overall_reciprocity(G) if G.number_of_edges() else 0.0

    # Degree assortativity
    try:
        report["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        report["degree_assortativity"] = float("nan")

    return report


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Community detection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """
    Return a nodeâ†’community_id mapping.
    Uses the Louvain method (via greedy modularity on undirected graph).
    """
    U = G.to_undirected()
    # Remove nodes with zero degree for better community detection
    U_pruned = U.copy()
    isolates = list(nx.isolates(U_pruned))
    U_pruned.remove_nodes_from(isolates)

    communities: dict[str, int] = {}
    if U_pruned.number_of_nodes() > 1:
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities,
            )
            comms = list(greedy_modularity_communities(U_pruned, weight="weight"))
            for cid, comm in enumerate(comms):
                for node in comm:
                    communities[node] = cid
        except Exception:
            pass

    # Assign isolates their own community
    base = max(communities.values(), default=-1) + 1
    for i, node in enumerate(isolates):
        communities[node] = base + i

    return communities


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Plotting helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

BG       = "white"
GRID_COL = "#dddddd"
TEXT_COL = "#111111"
ACCENT   = "#2563eb"
PALETTE  = [
    "#2563eb","#16a34a","#dc2626","#7c3aed","#ea580c",
    "#0891b2","#65a30d","#db2777","#9333ea","#b45309",
]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#aaaaaa",
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "legend.facecolor":  "#f8f8f8",
    "legend.edgecolor":  "#cccccc",
})


def _shorten(fqn: str, max_len: int = 28) -> str:
    """Display a short class name: keep simple name + one parent package."""
    parts = fqn.split(".")
    short = ".".join(parts[-2:]) if len(parts) >= 2 else fqn
    return short if len(short) <= max_len else short[:max_len - 1] + "..."


def _bar_chart(ax, labels, values, title, color=ACCENT, horizontal=True):
    labels  = [_shorten(l) for l in labels]
    if horizontal:
        bars = ax.barh(labels, values, color=color, alpha=0.85,
                       edgecolor="#00000015", linewidth=0.5)
        ax.set_xlabel("Score")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left",
                    fontsize=8, color=TEXT_COL)
    else:
        bars = ax.bar(labels, values, color=color, alpha=0.85,
                      edgecolor="#00000015", linewidth=0.5)
        ax.set_ylabel("Score")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Individual plot routines
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_graph_overview(G: nx.DiGraph, communities: dict[str, int],
                        repo_name: str, out_dir: Path):
    """Spring-layout graph coloured by community."""
    print("  Plotting graph overview â€¦")
    n = G.number_of_nodes()

    # For large graphs, sample top-N nodes by betweenness
    MAX_SHOW = 120
    if n > MAX_SHOW:
        bc = nx.betweenness_centrality(G, normalized=True)
        top = sorted(bc, key=bc.get, reverse=True)[:MAX_SHOW]
        sub = G.subgraph(top)
    else:
        sub = G

    weights = np.array([d.get("weight", 1.0) for _, _, d in sub.edges(data=True)])
    w_norm  = (weights / weights.max()) if len(weights) else weights

    node_list = list(sub.nodes())
    n_comms   = max(communities.values(), default=0) + 1
    cmap      = plt.colormaps.get_cmap("tab20")
    node_colors = [cmap(communities.get(nd, 0) % 20 / 20) for nd in node_list]

    deg = dict(sub.degree())
    node_sizes = [30 + 200 * (deg.get(n, 0) / max(deg.values(), default=1)) for n in node_list]

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    print("    computing layout â€¦")
    seed = 42
    pos = nx.spring_layout(sub, seed=seed, k=2.5 / math.sqrt(max(sub.number_of_nodes(), 1)),
                            weight="weight", iterations=60)

    # Edges
    nx.draw_networkx_edges(
        sub, pos, ax=ax,
        width=0.4 + 1.4 * w_norm,
        edge_color=[cm.Blues(0.3 + 0.5 * ww) for ww in w_norm],
        alpha=0.55,
        arrows=True, arrowsize=6,
        connectionstyle="arc3,rad=0.08",
    )
    # Nodes
    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=0.4,
        edgecolors="#00000030",
    )
    # Labels for hub nodes only
    bc = nx.betweenness_centrality(sub, normalized=True)
    thresh = sorted(bc.values(), reverse=True)[min(15, len(bc) - 1)]
    labels = {n: _shorten(n, 18) for n in sub.nodes() if bc[n] >= thresh}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax,
                            font_size=7, font_color=TEXT_COL)

    # Legend patches for communities
    unique_comms = sorted(set(communities.get(n, 0) for n in node_list))
    legend_patches = [
        mpatches.Patch(color=cmap(c % 20 / 20), label=f"Module {c}")
        for c in unique_comms[:10]
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              title="Communities", title_fontsize=9)

    title = f"{repo_name}  â€”  Dependency Graph\n"
    if n > MAX_SHOW:
        title += f"(top {MAX_SHOW} nodes by betweenness, {G.number_of_nodes()} total)"
    ax.set_title(title, fontsize=14, color=TEXT_COL, pad=12)
    ax.axis("off")
    plt.tight_layout()
    out = out_dir / "01_graph_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_degree_distribution(G: nx.DiGraph, out_dir: Path):
    """In-degree and out-degree distributions (log-log)."""
    print("  Plotting degree distribution â€¦")
    in_deg  = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, vals, label, col in [
        (axes[0], in_deg,  "In-Degree",  "#58a6ff"),
        (axes[1], out_deg, "Out-Degree", "#3fb950"),
    ]:
        unique, counts = np.unique(vals, return_counts=True)
        ax.scatter(unique, counts, color=col, s=40, alpha=0.8, edgecolors="none")
        ax.set_xlabel("Degree k")
        ax.set_ylabel("Count P(k)")
        ax.set_title(f"{label} Distribution")
        ax.set_xscale("log" if unique.max() > 10 else "linear")
        ax.set_yscale("log" if counts.max() > 5 else "linear")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Power-law fit annotation
        if len(unique) > 3 and unique.max() > 5:
            try:
                mask = (unique > 0) & (counts > 0)
                coeffs = np.polyfit(np.log(unique[mask]), np.log(counts[mask]), 1)
                ax.text(0.62, 0.85, f"slope ~= {coeffs[0]:.2f}",
                        transform=ax.transAxes, fontsize=10, color=col)
            except Exception:
                pass

    fig.suptitle("Degree Distributions", fontsize=14, color=TEXT_COL)
    plt.tight_layout()
    out = out_dir / "02_degree_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_top_centralities(df: pd.DataFrame, out_dir: Path, top_n: int = 15):
    """Top-N bar charts for the main centrality measures."""
    print("  Plotting top centralities â€¦")
    measures = [
        ("betweenness_centrality", "#ff7b72",  "Betweenness Centrality"),
        ("pagerank",               "#ffa657",  "PageRank"),
        ("in_degree_centrality",   "#58a6ff",  "In-Degree Centrality"),
        ("out_degree_centrality",  "#3fb950",  "Out-Degree Centrality"),
        ("closeness_centrality",   "#d2a8ff",  "Closeness Centrality"),
        ("eigenvector_centrality", "#79c0ff",  "Eigenvector Centrality"),
        ("katz_centrality",        "#56d364",  "Katz Centrality"),
        ("hub_score",              "#ffa657",  "HITS Hub Score"),
        ("authority_score",        "#f47067",  "HITS Authority Score"),
        ("harmonic_centrality",    "#c9a0ff",  "Harmonic Centrality"),
    ]

    n_plots = len(measures)
    cols    = 2
    rows    = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.2))
    axes_flat = axes.flatten()

    for i, (col_name, color, title) in enumerate(measures):
        ax = axes_flat[i]
        if col_name not in df.columns:
            ax.set_visible(False)
            continue
        top = df[col_name].nlargest(top_n)
        _bar_chart(ax, list(top.index), list(top.values), title, color=color)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Top-{top_n} Nodes by Centrality Measure", fontsize=15,
                 color=TEXT_COL, y=1.01)
    plt.tight_layout()
    out = out_dir / "03_top_centralities.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_centrality_correlation(df: pd.DataFrame, out_dir: Path):
    """Heatmap of Pearson correlations between centrality measures."""
    print("  Plotting centrality correlation heatmap â€¦")
    cols = [
        "betweenness_centrality", "pagerank", "in_degree_centrality",
        "out_degree_centrality",  "closeness_centrality",
        "eigenvector_centrality", "katz_centrality",
        "hub_score", "authority_score", "harmonic_centrality",
        "clustering", "betweenness_weighted",
    ]
    cols_avail = [c for c in cols if c in df.columns]
    corr = df[cols_avail].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = plt.colormaps.get_cmap("RdYlBu_r")
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    tick_labels = [c.replace("_", "\n") for c in cols_avail]
    ax.set_xticks(range(len(cols_avail)))
    ax.set_yticks(range(len(cols_avail)))
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=8)

    for i in range(len(cols_avail)):
        for j in range(len(cols_avail)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else TEXT_COL)

    ax.set_title("Centrality Measure Correlation Matrix", fontsize=14, color=TEXT_COL)
    plt.tight_layout()
    out = out_dir / "04_centrality_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_pagerank_vs_betweenness(df: pd.DataFrame, out_dir: Path):
    """Scatter: PageRank vs Betweenness, sized by in-degree."""
    print("  Plotting PageRank vs Betweenness scatter â€¦")
    fig, ax = plt.subplots(figsize=(12, 8))

    pr  = df["pagerank"].values
    bt  = df["betweenness_centrality"].values
    ind = df["in_degree"].values.astype(float)

    sc = ax.scatter(bt, pr, c=ind, s=30 + 250 * (pr / pr.max() if pr.max() else 1),
                    cmap="plasma", alpha=0.75, edgecolors="#00000020", linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("In-Degree", color=TEXT_COL)

    # Annotate top outliers
    thresh_pr = np.percentile(pr, 93)
    thresh_bt = np.percentile(bt, 93)
    for i, (x, y) in enumerate(zip(bt, pr)):
        if x >= thresh_bt or y >= thresh_pr:
            ax.annotate(_shorten(df.index[i], 22), (x, y),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=7, color=TEXT_COL, alpha=0.9)

    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("PageRank")
    ax.set_title("PageRank  vs  Betweenness Centrality\n(dot size ~ PageRank, colour ~ in-degree)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / "05_pagerank_vs_betweenness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_community_sizes(communities: dict[str, int], out_dir: Path):
    """Bar chart of community sizes."""
    print("  Plotting community sizes â€¦")
    from collections import Counter
    sizes = Counter(communities.values())
    sorted_sizes = sorted(sizes.items(), key=lambda x: -x[1])
    labels = [f"Module {k}" for k, _ in sorted_sizes[:30]]
    values = [v for _, v in sorted_sizes[:30]]

    cmap   = plt.colormaps.get_cmap("tab20")
    colors = [cmap(i % 20 / 20) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.9,
                  edgecolor="#00000015", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    ax.set_xlabel("Community / Module")
    ax.set_ylabel("Number of Classes")
    ax.set_title("Community Sizes  (Greedy Modularity)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / "06_community_sizes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_edge_weight_distribution(G: nx.DiGraph, out_dir: Path):
    """Histogram of edge weights."""
    print("  Plotting edge weight distribution â€¦")
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    if not weights:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, log_scale, suffix in [(axes[0], False, ""), (axes[1], True, " (log scale)")]:
        ax.hist(weights, bins=40, color=ACCENT, alpha=0.8,
                edgecolor="#00000012", linewidth=0.5, log=log_scale)
        ax.axvline(np.mean(weights), color="#ff7b72", lw=1.5, ls="--",
                   label=f"mean={np.mean(weights):.2f}")
        ax.axvline(np.median(weights), color="#3fb950", lw=1.5, ls="--",
                   label=f"median={np.median(weights):.2f}")
        ax.legend(fontsize=9)
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Count")
        ax.set_title(f"Edge Weight Distribution{suffix}")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "07_edge_weight_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_hubs_authorities(df: pd.DataFrame, out_dir: Path, top_n: int = 15):
    """Side-by-side hub vs authority scores."""
    print("  Plotting hub/authority chart â€¦")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, color, title in [
        (axes[0], "hub_score",       "#ffa657", "Top Hub Nodes (fan-out)"),
        (axes[1], "authority_score", "#d2a8ff", "Top Authority Nodes (fan-in)"),
    ]:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        top = df[col].nlargest(top_n)
        _bar_chart(ax, list(top.index), list(top.values), title, color=color)

    fig.suptitle("HITS Algorithm  â€”  Hubs & Authorities", fontsize=14, color=TEXT_COL)
    plt.tight_layout()
    out = out_dir / "08_hubs_authorities.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_summary_dashboard(report: dict, df: pd.DataFrame,
                           repo_name: str, out_dir: Path):
    """Multi-panel summary dashboard."""
    print("  Plotting summary dashboard â€¦")
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

    # â”€â”€ 1. Topology text box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.axis("off")
    lines = [
        f"Repository:   {repo_name}",
        f"Nodes (classes):  {report['nodes']:,}",
        f"Edges (deps):     {report['edges']:,}",
        f"Graph Density:    {report['density']:.4f}",
        f"Avg Edge Weight:  {report['avg_weight']:.2f}",
        f"Weak Components:  {report['weakly_connected_components']}",
        f"Strong Components:{report['strongly_connected_components']}",
        f"Largest WCC:       {report['largest_wcc_size']} nodes",
        f"Is DAG:            {report['is_dag']}",
        f"Reciprocity:       {report['reciprocity']:.3f}",
        f"Avg Clustering:    {report['avg_clustering']:.3f}",
        f"Degree Assort.:    {report.get('degree_assortativity', 'N/A')}",
        f"Diameter (WCC):    {report.get('diameter', 'N/A')}",
        f"Avg Shortest Path: {report.get('avg_shortest_path', 'N/A')}",
    ]
    ax0.text(0.03, 0.97, "\n".join(lines), transform=ax0.transAxes,
             va="top", ha="left", fontsize=10,
             fontfamily="monospace", color=TEXT_COL,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                       edgecolor=ACCENT, linewidth=1.2))
    ax0.set_title("Graph Topology Report", fontsize=12, color=TEXT_COL, pad=6)

    # â”€â”€ 2. Betweenness top-10 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax1 = fig.add_subplot(gs[0, 2:])
    if "betweenness_centrality" in df.columns:
        top = df["betweenness_centrality"].nlargest(10)
        _bar_chart(ax1, list(top.index), list(top.values),
                   "Top-10 Betweenness Centrality", color="#ff7b72")

    # â”€â”€ 3. In-degree distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax2 = fig.add_subplot(gs[1, :2])
    in_degs = df["in_degree"].values if "in_degree" in df.columns else []
    if len(in_degs):
        ax2.hist(in_degs, bins=30, color="#58a6ff", alpha=0.85,
                 edgecolor="#00000012")
        ax2.set_xlabel("In-Degree")
        ax2.set_ylabel("Count")
        ax2.set_title("In-Degree Distribution")
        ax2.grid(alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

    # â”€â”€ 4. PageRank distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax3 = fig.add_subplot(gs[1, 2:])
    if "pagerank" in df.columns:
        ax3.hist(df["pagerank"].values, bins=30, color="#3fb950", alpha=0.85,
                 edgecolor="#00000012")
        ax3.set_xlabel("PageRank")
        ax3.set_ylabel("Count")
        ax3.set_title("PageRank Distribution")
        ax3.grid(alpha=0.3)
        ax3.spines[["top", "right"]].set_visible(False)

    # â”€â”€ 5. Clustering distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax4 = fig.add_subplot(gs[2, :2])
    if "clustering" in df.columns:
        cl_vals = df["clustering"].values
        ax4.hist(cl_vals, bins=30, color="#d2a8ff", alpha=0.85,
                 edgecolor="#00000012")
        ax4.set_xlabel("Clustering Coefficient")
        ax4.set_ylabel("Count")
        ax4.set_title("Clustering Coefficient Distribution")
        ax4.grid(alpha=0.3)
        ax4.spines[["top", "right"]].set_visible(False)

    # â”€â”€ 6. PageRank top-10 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax5 = fig.add_subplot(gs[2, 2:])
    if "pagerank" in df.columns:
        top = df["pagerank"].nlargest(10)
        _bar_chart(ax5, list(top.index), list(top.values),
                   "Top-10 PageRank", color="#ffa657")

    fig.suptitle(f"Graph Analysis Dashboard  â€”  {repo_name}",
                 fontsize=16, color=TEXT_COL, y=1.01)
    plt.savefig(out_dir / "00_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out_dir / '00_dashboard.png'}")


def plot_package_heatmap(G: nx.DiGraph, out_dir: Path, depth: int = 2):
    """
    Aggregate classes to package level (first `depth` segments of FQN),
    then show a heatmap of inter-package coupling counts.
    """
    print("  Plotting package dependency heatmap â€¦")

    def pkg(fqn: str) -> str:
        parts = fqn.split(".")
        return ".".join(parts[:depth]) if len(parts) >= depth else fqn

    pkg_edges: dict[tuple[str, str], float] = defaultdict(float)
    for u, v, d in G.edges(data=True):
        pu, pv = pkg(u), pkg(v)
        if pu != pv:
            pkg_edges[(pu, pv)] += d.get("weight", 1.0)

    all_pkgs = sorted(set(p for pair in pkg_edges for p in pair))
    if len(all_pkgs) < 2:
        print("    Not enough packages to plot heatmap, skipping.")
        return

    idx = {p: i for i, p in enumerate(all_pkgs)}
    mat = np.zeros((len(all_pkgs), len(all_pkgs)))
    for (pu, pv), w in pkg_edges.items():
        mat[idx[pu], idx[pv]] = w

    fig, ax = plt.subplots(figsize=(min(18, len(all_pkgs) + 3),
                                    min(16, len(all_pkgs) + 2)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Coupling Weight", color=TEXT_COL)

    tick_labels = [_shorten(p, 22) for p in all_pkgs]
    ax.set_xticks(range(len(all_pkgs)))
    ax.set_yticks(range(len(all_pkgs)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(f"Package-Level Coupling Heatmap (depth={depth})", fontsize=13)
    plt.tight_layout()
    out = out_dir / "09_package_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


def plot_scc_analysis(G: nx.DiGraph, out_dir: Path):
    """Visualise SCCs as a condensation DAG."""
    print("  Plotting SCC condensation â€¦")
    sccs = list(nx.strongly_connected_components(G))
    sccs_sorted = sorted(sccs, key=len, reverse=True)
    top_sccs = sccs_sorted[:min(20, len(sccs_sorted))]
    sizes = [len(s) for s in top_sccs]
    labels = [f"SCC {i + 1}\n(n={sz})" for i, sz in enumerate(sizes)]

    cmap = plt.colormaps.get_cmap("plasma")
    colors = [cmap(i / max(len(top_sccs) - 1, 1)) for i in range(len(top_sccs))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of top SCC sizes
    ax0 = axes[0]
    bars = ax0.bar(labels, sizes, color=colors, alpha=0.85,
                   edgecolor="#00000015", linewidth=0.5)
    for bar, sz in zip(bars, sizes):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(sz), ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    ax0.set_xlabel("Strongly Connected Component")
    ax0.set_ylabel("Size (# classes)")
    ax0.set_title(f"Top-{len(top_sccs)} SCCs by Size")
    plt.setp(ax0.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)
    ax0.spines[["top", "right"]].set_visible(False)

    # Pie chart
    ax1 = axes[1]
    pie_sizes = sizes[:8] + ([sum(sizes[8:])] if len(sizes) > 8 else [])
    pie_labels = labels[:8] + (["Others"] if len(sizes) > 8 else [])
    pie_colors = colors[:8] + (["#555555"] if len(sizes) > 8 else [])
    ax1.pie(pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"color": TEXT_COL, "fontsize": 8},
            wedgeprops={"edgecolor": BG, "linewidth": 1.5})
    ax1.set_title("SCC Size Distribution (pie)")

    fig.suptitle("Strongly Connected Components Analysis", fontsize=14, color=TEXT_COL)
    plt.tight_layout()
    out = out_dir / "10_scc_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"    saved {out}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Textual report
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_report(report: dict, df: pd.DataFrame, communities: dict, repo_name: str):
    sep  = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print(f"  DEPENDENCY GRAPH ANALYSIS  --  {repo_name}")
    print(sep)

    print("\n[TOPOLOGY SUMMARY]")
    print(thin)
    for k, v in report.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:<34} {v:.4f}")
        else:
            print(f"  {label:<34} {v}")

    print(f"\n[TOP-10 NODES BY KEY CENTRALITY MEASURES]")
    print(thin)
    for col, title in [
        ("betweenness_centrality", "Betweenness"),
        ("pagerank",               "PageRank"),
        ("in_degree_centrality",   "In-Degree"),
        ("out_degree_centrality",  "Out-Degree"),
        ("eigenvector_centrality", "Eigenvector"),
        ("hub_score",              "Hub (HITS)"),
        ("authority_score",        "Authority (HITS)"),
        ("katz_centrality",        "Katz"),
        ("harmonic_centrality",    "Harmonic"),
    ]:
        if col not in df.columns:
            continue
        print(f"\n  >> {title}")
        top = df[col].nlargest(10)
        for rank, (node, val) in enumerate(top.items(), 1):
            print(f"    {rank:2d}. {_shorten(node, 50):<52} {val:.6f}")

    n_comms = len(set(communities.values()))
    print(f"\n[COMMUNITY DETECTION]")
    print(thin)
    print(f"  Detected {n_comms} communities (greedy modularity)")
    from collections import Counter
    sizes = Counter(communities.values())
    for cid, sz in sorted(sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  Module {cid:3d}: {sz:4d} classes")

    print(f"\n{sep}\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

REPOS = ["acmeair", "jpetstore-6", "sample.daytrader7", "sample.plantsbywebsphere"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and analyse a Java dependency graph from a repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Available repos:
              {chr(10).join('  ' + r for r in REPOS)}
        """),
    )
    p.add_argument("--repo",   default="jpetstore-6",
                   choices=REPOS, help="Which repo to analyse (default: jpetstore-6)")
    p.add_argument("--top-n",  type=int, default=15,
                   help="Number of top nodes shown in bar charts (default: 15)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful for headless environments)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

    repo_name = args.repo
    repo_path = _DATA / repo_name
    if not repo_path.exists():
        sys.exit(f"[ERROR] Repo not found: {repo_path}")

    out_dir = _OUT / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Analysing: {repo_name}")
    print(f"  Repo path: {repo_path}")
    print(f"  Output:    {out_dir}")
    print(f"{'=' * 60}\n")

    # -- 1. Build graph -------------------------------------------------------
    print("[Step 1/5] Building dependency graph...")
    G = build_graph_from_repo(repo_path, filter_external=False)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")

    if G.number_of_nodes() == 0:
        sys.exit("[ERROR] Graph has no nodes - check that the repo contains .java files.")

    # -- 2. Topology report ---------------------------------------------------
    print("[Step 2/5] Computing topology metrics...")
    report = topology_report(G)

    # -- 3. Centralities ------------------------------------------------------
    print("\n[Step 3/5] Computing centralities...")
    df = compute_centralities(G)
    csv_path = out_dir / "centrality_scores.csv"
    df.to_csv(csv_path)
    print(f"  Centrality data saved to {csv_path}\n")

    # -- 4. Community detection -----------------------------------------------
    print("[Step 4/5] Detecting communities...")
    communities = detect_communities(G)

    # -- 5. Print textual report ----------------------------------------------
    print_report(report, df, communities, repo_name)

    # -- 6. Visualisations ----------------------------------------------------
    print("[Step 5/5] Generating visualisations...")

    if args.no_show:
        plt.ioff()

    plot_summary_dashboard(report, df, repo_name, out_dir)
    plot_graph_overview(G, communities, repo_name, out_dir)
    plot_degree_distribution(G, out_dir)
    plot_top_centralities(df, out_dir, top_n=args.top_n)
    plot_centrality_correlation(df, out_dir)
    plot_pagerank_vs_betweenness(df, out_dir)
    plot_community_sizes(communities, out_dir)
    plot_edge_weight_distribution(G, out_dir)
    plot_hubs_authorities(df, out_dir, top_n=args.top_n)
    plot_package_heatmap(G, out_dir, depth=2)
    plot_scc_analysis(G, out_dir)

    print(f"\n[DONE]  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

