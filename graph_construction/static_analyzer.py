"""
Phase 1 — Static Analysis Ingestion
====================================
Parses java-callgraph output (.txt) and Jarviz dependency exports (.jsonl)
into a unified edge list suitable for graph construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rich.console import Console

console = Console()


@dataclass
class RawEdge:
    """A directed dependency edge extracted from static analysis."""

    source: str  # fully-qualified class name
    target: str  # fully-qualified class name
    kind: str    # "call", "extend", "implement", "field", "import"
    weight: float = 1.0


# ---------------------------------------------------------------------------
# java-callgraph parser
# ---------------------------------------------------------------------------

# java-callgraph output line format:
#   M:com.example.Foo:bar(I)V  (M)com.example.Bar:baz()V
_CALLGRAPH_RE = re.compile(
    r"^M:(?P<src_cls>[^:]+):(?P<src_mth>\S+)\s+\(M\)(?P<tgt_cls>[^:]+):(?P<tgt_mth>\S+)$"
)


def parse_java_callgraph(path: str | Path) -> list[RawEdge]:
    """
    Parse a java-callgraph output file.

    Each non-empty, non-comment line that matches the method-call format
    produces one RawEdge with kind='call'.
    """
    edges: list[RawEdge] = []
    path = Path(path)
    if not path.exists():
        console.print(f"[yellow]Warning:[/] callgraph file not found: {path}")
        return edges

    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _CALLGRAPH_RE.match(line)
            if m:
                src, tgt = m.group("src_cls"), m.group("tgt_cls")
                if src != tgt:  # skip self-loops at class level
                    edges.append(RawEdge(source=src, target=tgt, kind="call"))
            else:
                console.print(f"[dim]Skipping malformed line {lineno}: {line[:80]}[/]")

    console.print(f"[green]java-callgraph:[/] parsed {len(edges):,} call edges from {path.name}")
    return edges


# ---------------------------------------------------------------------------
# Jarviz .jsonl parser
# ---------------------------------------------------------------------------

def parse_jarviz_jsonl(path: str | Path) -> list[RawEdge]:
    """
    Parse a Jarviz JSON-Lines export file.

    Jarviz emits one JSON object per line.  The expected schema (all strings):
        {
          "appSetName": "...",
          "artifactGroup": "...",
          "artifactName": "...",
          "artifactVersion": "...",
          "artifactFilter": "...",
          "className": "com.example.Foo",
          "classDependency": "com.example.Bar",
          "methodName": "doSomething",
          "methodDependency": "process"
        }
    """
    edges: list[RawEdge] = []
    path = Path(path)
    if not path.exists():
        console.print(f"[yellow]Warning:[/] Jarviz file not found: {path}")
        return edges

    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                src = obj.get("className", "").strip()
                tgt = obj.get("classDependency", "").strip()
                if src and tgt and src != tgt:
                    edges.append(RawEdge(source=src, target=tgt, kind="call"))
            except json.JSONDecodeError as exc:
                console.print(f"[yellow]Jarviz parse error line {lineno}:[/] {exc}")

    console.print(f"[green]Jarviz:[/] parsed {len(edges):,} dependency edges from {path.name}")
    return edges


# ---------------------------------------------------------------------------
# ObjectAid .ucls XML parser (bonus)
# ---------------------------------------------------------------------------

def parse_objectaid_ucls(path: str | Path) -> list[RawEdge]:
    """
    Parse ObjectAid .ucls XML exports.

    Relationship types extracted:
        generalization  -> 'extend'
        realization     -> 'implement'
        dependency      -> 'call'
        association     -> 'field'
    """
    import xml.etree.ElementTree as ET

    KIND_MAP = {
        "generalization": "extend",
        "realization": "implement",
        "dependency": "call",
        "association": "field",
    }

    edges: list[RawEdge] = []
    path = Path(path)
    if not path.exists():
        console.print(f"[yellow]Warning:[/] ObjectAid file not found: {path}")
        return edges

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        console.print(f"[red]ObjectAid XML parse error:[/] {exc}")
        return edges

    # Build id -> className map from <element> nodes
    id_to_class: dict[str, str] = {}
    for el in root.iter("element"):
        eid = el.get("id") or el.get("xmi:id")
        name = el.get("qualifiedName") or el.get("name")
        if eid and name:
            id_to_class[eid] = name

    for rel in root.iter("relationship"):
        src_id = rel.get("source")
        tgt_id = rel.get("target")
        kind_raw = rel.get("xmi:type", "dependency").lower()
        kind = KIND_MAP.get(kind_raw, "call")
        src = id_to_class.get(src_id or "", "")
        tgt = id_to_class.get(tgt_id or "", "")
        if src and tgt and src != tgt:
            edges.append(RawEdge(source=src, target=tgt, kind=kind))

    console.print(f"[green]ObjectAid:[/] parsed {len(edges):,} relationship edges from {path.name}")
    return edges
