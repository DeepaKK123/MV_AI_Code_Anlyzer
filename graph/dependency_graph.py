"""
graph/dependency_graph.py
NetworkX call-dependency graph builder and traversal engine.

- build_graph(): scans all .B files, creates directed graph of CALL relationships
- get_impact(): given a subroutine name, returns all callers, callees, files,
  and risk flags
- save_graph() / load_graph(): persist graph to JSON for fast restarts

Usage (standalone test):
    python -m graph.dependency_graph ./unidata_source
"""

import json
from pathlib import Path
import networkx as nx
from parser.unidata_parser import parse_unidata_file


def build_graph(source_dir: str) -> nx.DiGraph:
    """
    Scan all .B and .bas files in source_dir (recursive) and build a
    directed call-dependency graph.
    Each node = subroutine name (uppercased).
    Each edge = CALL relationship (caller -> callee).
    Node attributes store file path and risk metadata.
    """
    G = nx.DiGraph()
    source_path = Path(source_dir)

    # Collect all source files
    all_files = list(source_path.rglob("*.B")) + list(source_path.rglob("*.bas"))

    if not all_files:
        print(f"  WARNING: No .B or .bas files found in {source_dir}")
        return G

    parsed = {}
    for f in all_files:
        try:
            info = parse_unidata_file(str(f))
            key = info.name.upper()
            parsed[key] = info
            G.add_node(key, **{
                "file_path": info.file_path,
                "opens": info.opens,
                "readu_files": info.readu_files,
                "unclosed": info.unclosed,
                "loop_lines": info.loops,
                "reads": info.reads,
                "writes": info.writes,
            })
        except Exception as e:
            print(f"  WARNING: Could not parse {f}: {e}")

    # Add directed CALL edges
    for name, info in parsed.items():
        for called in info.calls:
            callee_key = called.upper()
            # Add callee node even if its file wasn't found (external reference)
            if callee_key not in G:
                G.add_node(callee_key, file_path="EXTERNAL", opens=[],
                           readu_files=[], unclosed=False, loop_lines=[],
                           reads=[], writes=[])
            G.add_edge(name, callee_key, relation="calls")

    return G


def get_impact(G: nx.DiGraph, subroutine: str) -> dict:
    """
    Return full impact analysis for a named subroutine:
    - direct_callers: subroutines that directly call this one
    - all_callers: entire upstream ancestor set
    - calls_into: everything this subroutine calls (descendants)
    - files_accessed: union of all file handles opened in the subtree
    - risk_flags: unclosed handles, locked reads, loop line numbers
    """
    name = subroutine.upper()

    if name not in G:
        return {"error": f"'{subroutine}' not found in dependency graph. "
                         f"Check the subroutine name or re-run setup.py."}

    callers = list(nx.ancestors(G, name))
    callees = list(nx.descendants(G, name))
    node_data = G.nodes[name]

    # Collect all file handles opened in the subroutine and its callees
    affected_files = set(node_data.get("opens", []))
    for c in callees:
        affected_files.update(G.nodes[c].get("opens", []))

    return {
        "target": name,
        "file_path": node_data.get("file_path", "unknown"),
        "direct_callers": list(G.predecessors(name)),
        "all_callers": callers,
        "calls_into": callees,
        "files_accessed": list(affected_files),
        "risk_flags": {
            "unclosed_file_handles": node_data.get("unclosed", False),
            "loop_lines": node_data.get("loop_lines", []),
            "locked_reads_readu": node_data.get("readu_files", []),
        },
        "summary": {
            "total_upstream_callers": len(callers),
            "total_downstream_callees": len(callees),
            "files_touched": len(affected_files),
        }
    }


def save_graph(G: nx.DiGraph, path: str):
    """Serialise graph to JSON (survives Python restarts)."""
    with open(path, "w") as f:
        json.dump(nx.node_link_data(G), f, indent=2)
    print(f"  Graph saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {path}")


def load_graph(path: str) -> nx.DiGraph:
    """Load graph from JSON file."""
    with open(path) as f:
        return nx.node_link_graph(json.load(f))


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "./unidata_source"
    G = build_graph(source)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print("Subroutines found:", list(G.nodes)[:20])
