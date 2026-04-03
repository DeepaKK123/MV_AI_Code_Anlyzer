"""
setup.py
One-time setup script. Run this BEFORE launching app.py.

What it does:
  1. Scans mv_source/ for MV BASIC files (any name, any extension) and builds the call dependency graph
  2. Saves the graph to graph.json
  3. Ingests all MV BASIC and .txt files into ChromaDB (embedding with nomic-embed-text)

Re-run whenever you add/change source files significantly.

Usage:
    python setup.py
"""

import sys
from pathlib import Path
from graph.dependency_graph import build_graph, save_graph
from rag.ingest import ingest_corpus

SOURCE_DIR = "./mv_source"    # Place your MV BASIC files here (any name/extension)
DOCS_DIR = "./documents"      # Place .txt contract/schema/Layout documents here
GRAPH_PATH = "./graph.json"
CHROMA_PATH = "./chroma_db"


def check_prerequisites():
    """Verify folders exist and have content."""
    errors = []
    warnings = []

    for folder in [SOURCE_DIR, DOCS_DIR]:
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)
            warnings.append(f"Created empty folder: {folder}")

    # Accept ALL files in mv_source regardless of extension
    # MV files like ORD.PROCESS have .PROCESS as suffix — not empty string
    source_path = Path(SOURCE_DIR)
    all_files = [f for f in source_path.rglob("*") if f.is_file()]

    if not all_files:
        errors.append(
            f"No source files found in {SOURCE_DIR}\n"
            "  -> Copy your MV BASIC source files into this folder."
        )

    for w in warnings:
        print(f"  ⚠  {w}")

    if errors:
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)

    return all_files


def main():
    print("=" * 60)
    print("  MultiValue AI POC — Setup")
    print("=" * 60)

    print("\n[1/3] Checking prerequisites...")
    all_files = check_prerequisites()
    print(f"  ✓  Found {len(all_files)} source files")
    for f in all_files:
        print(f"      {f.name}")

    print("\n[2/3] Building dependency graph...")
    G = build_graph(SOURCE_DIR)
    save_graph(G, GRAPH_PATH)
    print(f"  ✓  Graph: {G.number_of_nodes()} subroutines, {G.number_of_edges()} call edges")

    print("\n[3/3] Ingesting corpus into ChromaDB RAG store...")
    try:
        ingest_corpus(SOURCE_DIR, DOCS_DIR, chroma_path=CHROMA_PATH)
        print("  ✓  Ingestion complete")
    except RuntimeError as e:
        print(f"  ✗  {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Start the app with:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()