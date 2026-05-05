"""
Phase 1 — Dynamic Execution Trace Parser
=========================================
Ingests runtime call traces (OpenTelemetry JSON or Mono2Micro format) and
produces weighted RawEdge objects reflecting actual production call frequency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich.console import Console

console = Console()


@dataclass
class TraceEdge:
    """A class-to-class call observed at runtime, with frequency count."""

    source: str
    target: str
    count: int   # number of times this call was observed across all traces
    use_cases: set[str] = None  # which business use-cases triggered this edge

    def __post_init__(self):
        if self.use_cases is None:
            self.use_cases = set()


# ---------------------------------------------------------------------------
# OpenTelemetry / generic JSON trace parser
# ---------------------------------------------------------------------------

def parse_otel_traces(path: str | Path) -> list[TraceEdge]:
    """
    Parse an OpenTelemetry-compatible JSON trace file.

    Expected schema (list of span objects):
        [
          {
            "traceId": "...",
            "spanId": "...",
            "parentSpanId": "...",     # optional; absent for root spans
            "operationName": "com.example.Foo/doSomething",
            "tags": {"http.url": "...", "db.type": "...", ...},
            "references": [...]
          },
          ...
        ]
    The class name is extracted from operationName (everything before the
    first '/') using Java fully-qualified class conventions.

    Parent→Child span pairs are interpreted as call edges.
    """
    path = Path(path)
    if not path.exists():
        console.print(f"[yellow]Warning:[/] OTel trace file not found: {path}")
        return []

    with path.open(encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            console.print(f"[red]OTel parse error:[/] {exc}")
            return []

    # Flatten nested trace/span structures
    spans: list[dict] = []
    if isinstance(data, list):
        for item in data:
            # Support both flat list-of-spans and {data: [{spans: [...]}]} formats
            if "spans" in item:
                spans.extend(item["spans"])
            elif "operationName" in item or "name" in item:
                spans.append(item)
            elif "data" in item:
                for trace in item.get("data", []):
                    spans.extend(trace.get("spans", []))

    span_id_to_class: dict[str, str] = {}
    for span in spans:
        sid = span.get("spanId") or span.get("spanID") or span.get("span_id")
        op = span.get("operationName") or span.get("name") or ""
        cls = op.split("/")[0].split("(")[0].strip()
        if sid and cls:
            span_id_to_class[sid] = cls

    counter: dict[tuple[str, str], int] = defaultdict(int)
    for span in spans:
        sid = span.get("spanId") or span.get("spanID") or span.get("span_id")
        pid = span.get("parentSpanId") or span.get("parentSpanID") or span.get("parent_span_id")
        if not pid:
            continue
        src = span_id_to_class.get(pid)
        tgt = span_id_to_class.get(sid)
        if src and tgt and src != tgt:
            counter[(src, tgt)] += 1

    edges = [
        TraceEdge(source=src, target=tgt, count=cnt)
        for (src, tgt), cnt in counter.items()
    ]
    console.print(f"[green]OTel traces:[/] extracted {len(edges):,} dynamic edges from {path.name}")
    return edges


# ---------------------------------------------------------------------------
# Mono2Micro execution trace parser
# ---------------------------------------------------------------------------

def parse_mono2micro_traces(path: str | Path) -> list[TraceEdge]:
    """
    Parse Mono2Micro's execution trace JSON format.

    Mono2Micro emits a list of use-case objects, each containing an ordered
    sequence of class interactions:
        [
          {
            "useCase": "BuyStock",
            "entryClass": "com.example.TradeAction",
            "trace": [
              "com.example.TradeAction",
              "com.example.TradeService",
              "com.example.AccountRepository",
              ...
            ]
          },
          ...
        ]
    Sequential pairs in the trace are treated as directed call edges.
    """
    path = Path(path)
    if not path.exists():
        console.print(f"[yellow]Warning:[/] Mono2Micro trace file not found: {path}")
        return []

    with path.open(encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Mono2Micro parse error:[/] {exc}")
            return []

    counter: dict[tuple[str, str], int] = defaultdict(int)
    use_case_map: dict[tuple[str, str], set[str]] = defaultdict(set)

    for uc_obj in data:
        uc_name = uc_obj.get("useCase", "unknown")
        trace: list[str] = uc_obj.get("trace", [])
        for i in range(len(trace) - 1):
            src, tgt = trace[i].strip(), trace[i + 1].strip()
            if src and tgt and src != tgt:
                counter[(src, tgt)] += 1
                use_case_map[(src, tgt)].add(uc_name)

    edges = [
        TraceEdge(source=src, target=tgt, count=cnt, use_cases=use_case_map[(src, tgt)])
        for (src, tgt), cnt in counter.items()
    ]
    console.print(
        f"[green]Mono2Micro traces:[/] extracted {len(edges):,} dynamic edges "
        f"from {len(data)} use-cases in {path.name}"
    )
    return edges
