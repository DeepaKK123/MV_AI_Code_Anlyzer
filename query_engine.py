"""
analysis/query_engine.py
Core orchestration engine.

Wires together:
  1. ChromaDB RAG retrieval (semantic similarity)
  2. NetworkX dependency graph (call chain traversal)
  3. Ollama LLM (qwen2.5-coder:14b) for plain-English analysis

Split into two methods so the Streamlit spinner works correctly:
  - prepare()  — does RAG + graph lookup eagerly (runs under spinner)
  - stream()   — streams LLM tokens one by one (runs after spinner)
  - analyse()  — convenience wrapper for CLI/testing use
"""

import json
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from graph.dependency_graph import get_impact, load_graph


# ── System prompt ─────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["context", "graph_context", "question"],
    template="""You are a MultiValue BASIC code analyst.
Analyse ONLY — never generate or suggest code changes.

SOURCE CODE:
{context}

GRAPH DATA:
{graph_context}

QUESTION:
{question}

Reply concisely covering:
1. What the code does (plain English)
2. Affected subroutines and files
3. Risk flags (unclosed handles, READU locks, LOOP risks)
4. Confidence: High / Medium / Low

Be specific and concise. No code suggestions.""",
)


# ── Quick reply patterns ───────────────────────────────────────────────────────
QUICK_REPLIES = {
    "hi": "Hi there! I'm your MultiValue Code Analyser. Ask me about any subroutine or paste a subroutine name in the sidebar to get started!",
    "hello": "Hello! Ready to analyse your MultiValue codebase. Type a subroutine name in the sidebar and ask me anything about it.",
    "hey": "Hey! MultiValue Assistant here. How can I help you analyse your codebase today?",
    "how are you": "I'm running great and ready to analyse your MultiValue code! What subroutine would you like to explore?",
    "what can you do": "I can help you with:\n\n1. **Code Explanation** — Ask what any subroutine does in plain English\n2. **Impact Analysis** — Ask what breaks if a subroutine changes\n3. **Risk Flagging** — Identify unclosed file handles, READU locks, and LOOP risks\n\nType a subroutine name in the sidebar and ask away!",
    "who are you": "I'm an AI-powered MultiValue code analysis assistant. I help developers understand legacy MV BASIC codebases — explaining subroutines, tracing call dependencies, and flagging risks.",
    "thanks": "You're welcome! Let me know if you have more questions about your codebase.",
    "thank you": "Happy to help! Ask me anything else about your MultiValue code.",
    "bye": "Goodbye! Come back anytime you need help analysing your MultiValue codebase.",
    "help": "I can help you with:\n\n1. **Code Explanation** — 'What does ORD.PROCESS do?'\n2. **Impact Analysis** — 'What is affected if I change INV.UPDATE?'\n3. **Risk Flags** — 'Are there any risks in ORD.PROCESS?'\n\nEnter a subroutine name in the sidebar to enable dependency graph analysis.",
}


def get_quick_reply(question: str):
    """Return instant reply for greetings and small talk, None for real queries."""
    cleaned = question.lower().strip().rstrip("!?.,")
    for pattern, reply in QUICK_REPLIES.items():
        if pattern in cleaned:
            return reply
    return None


# ── Main engine ───────────────────────────────────────────────────────────────
class MVAnalysisEngine:
    """
    Main analysis engine. Initialise once; call prepare() then stream() per question.

    Args:
        chroma_path: path to persisted ChromaDB vector store (from setup.py)
        graph_path:  path to graph.json (from setup.py)
        llm_model:   Ollama model name (default: qwen2.5-coder:14b)
        top_k:       number of RAG chunks to retrieve per query (default: 3)
    """

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        graph_path: str = "./graph.json",
        llm_model: str = "qwen2.5-coder:14b",
        top_k: int = 3,
    ):
        print(f"  Loading LLM: {llm_model} ...")
        self.llm = Ollama(
            model=llm_model,
            temperature=0,
            num_predict=512,
            num_ctx=2048,
        )

        print(f"  Loading embeddings: nomic-embed-text ...")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        print(f"  Loading ChromaDB from {chroma_path} ...")
        self.vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        print(f"  Loading dependency graph from {graph_path} ...")
        self.graph = load_graph(graph_path)
        print("  Engine ready.")

    def prepare(self, question: str, subroutine_name: str = None) -> dict:
        """
        Step 1 — Run RAG retrieval + graph lookup eagerly.
        This runs under the Streamlit spinner so user sees progress.

        Returns dict with keys:
            prompt  - fully assembled prompt string ready for the LLM
            sources - list of source file paths retrieved from RAG
            impact  - dependency graph data (empty dict if no subroutine given)
        """
        # Step 1: Semantic retrieval from RAG store
        relevant_docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(
            [f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
             for d in relevant_docs]
        )

        # Step 2: Dependency graph traversal
        graph_context = "No specific subroutine identified for graph traversal."
        impact = {}
        if subroutine_name and subroutine_name.strip():
            impact = get_impact(self.graph, subroutine_name.strip())
            graph_context = json.dumps(impact, indent=2)

        # Step 3: Assemble prompt — do NOT call LLM yet
        prompt = ANALYSIS_PROMPT.format(
            context=context,
            graph_context=graph_context,
            question=question,
        )

        return {
            "prompt": prompt,
            "sources": [d.metadata.get("source", "") for d in relevant_docs],
            "impact": impact,
        }

    def stream(self, prompt: str):
        """
        Step 2 — Stream LLM tokens one by one.
        Call this AFTER prepare(), outside the spinner.

        Yields string chunks as the LLM generates them.
        """
        for chunk in self.llm.stream(prompt):
            yield chunk

    def analyse(self, question: str, subroutine_name: str = None) -> dict:
        """
        Convenience method for CLI/testing — runs prepare() and collects
        full streamed response into a single string.

        Returns dict with keys:
            answer  - complete LLM response string
            sources - list of source file paths
            impact  - dependency graph data
        """
        quick = get_quick_reply(question)
        if quick:
            return {"answer": quick, "sources": [], "impact": {}}

        result = self.prepare(question, subroutine_name)
        full_response = "".join(self.stream(result["prompt"]))

        return {
            "answer": full_response,
            "sources": result["sources"],
            "impact": result["impact"],
        }
