"""
app.py
Streamlit developer chat interface for the UniData MV AI Code Analyser.

Run with:
    streamlit run app.py

The app caches the analysis engine on first load (takes ~10-20 seconds).
Subsequent queries use the cached engine.
"""

import streamlit as st
from pathlib import Path
from analysis.query_engine import MVAnalysisEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniData MV AI Analyser",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 UniData MV Code Analyser")
st.caption("Analysis and explanation only — no code changes are proposed.")

# ── Pre-flight check: required files must exist ──────────────────────────────
chroma_path = "./chroma_db"
graph_path = "./graph.json"

missing = []
if not Path(chroma_path).exists():
    missing.append("`chroma_db/` vector store")
if not Path(graph_path).exists():
    missing.append("`graph.json` dependency graph")

if missing:
    st.error(
        "⚠️ Setup not complete. The following are missing:\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nPlease run:\n```\npython setup.py\n```"
    )
    st.stop()


# ── Load engine (cached across sessions) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI analysis engine...")
def load_engine():
    return MVAnalysisEngine(
        chroma_path=chroma_path,
        graph_path=graph_path,
    )


engine = load_engine()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Query Options")

    subroutine = st.text_input(
        "Subroutine name (optional)",
        placeholder="e.g. ORD.PROCESS",
        help="Enter a subroutine name to enable dependency graph analysis and impact assessment.",
    )

    st.divider()
    st.markdown("**💡 Example questions:**")
    examples = [
        "What does ORD.PROCESS do?",
        "What files does INV.UPDATE touch?",
        "If I change ORD.PROCESS, what else is affected?",
        "Are there any risk flags in INV.UPDATE?",
        "What subroutines call CUST.VALIDATE?",
        "Explain the locking pattern in this routine.",
        "What business process does this code handle?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill_question"] = ex

    st.divider()
    st.markdown("**📊 Graph stats**")
    try:
        from graph.dependency_graph import load_graph
        G = load_graph(graph_path)
        st.metric("Subroutines", G.number_of_nodes())
        st.metric("Call relationships", G.number_of_edges())
    except Exception:
        st.caption("Graph not loaded")

    if st.button("🗑️ Clear chat history"):
        st.session_state["messages"] = []
        st.rerun()

# ── Chat session state ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("impact"):
            with st.expander("📈 Dependency graph data"):
                st.json(msg["impact"])
        if msg.get("sources"):
            with st.expander("📁 Source files referenced"):
                for s in set(msg["sources"]):
                    if s:
                        st.code(s)

# ── Chat input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask about your UniData codebase...") or prefill

if question:
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # Run analysis
    with st.chat_message("assistant"):
        with st.spinner("Analysing... (first query may take 15-40 seconds on CPU)"):
            try:
                result = engine.analyse(
                    question,
                    subroutine_name=subroutine.strip() if subroutine else None,
                )

                st.write(result["answer"])

                if result.get("impact") and "error" not in result["impact"]:
                    with st.expander("📈 Dependency graph data"):
                        st.json(result["impact"])

                if result.get("sources"):
                    with st.expander("📁 Source files referenced"):
                        for s in set(result["sources"]):
                            if s:
                                st.code(s)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "impact": result.get("impact", {}),
                    "sources": result.get("sources", []),
                })

            except Exception as e:
                error_msg = f"⚠️ Analysis error: {str(e)}\n\nCheck that Ollama is running (`ollama serve`) and models are pulled."
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                })
