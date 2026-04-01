"""
analysis/query_engine.py
Core orchestration engine.

Wires together:
  1. ChromaDB RAG retrieval (semantic similarity)
  2. NetworkX dependency graph (call chain traversal)
  3. Ollama LLM (DeepSeek Coder 13B) for plain-English analysis

The LLM prompt explicitly instructs the model to ANALYSE ONLY —
never generate or suggest code changes.

Usage:
    from analysis.query_engine import MVAnalysisEngine
    engine = MVAnalysisEngine(chroma_path="./chroma_db", graph_path="./graph.json")
    result = engine.analyse("What does ORD.PROCESS do?", subroutine_name="ORD.PROCESS")
    print(result["answer"])
"""

import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from graph.dependency_graph import get_impact, load_graph


# ── System prompt ────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["context", "graph_context", "question"],
    template="""You are an expert UniData MultiValue BASIC developer assistant.
Your job is ONLY to analyse and explain existing code.
You must NEVER generate, propose, or suggest code changes of any kind.

RELEVANT SOURCE CODE AND DOCUMENTS:
{context}

DEPENDENCY GRAPH DATA:
{graph_context}

DEVELOPER QUESTION:
{question}

Provide a clear, structured analysis covering:
1. Plain-English explanation of what the code does and its business purpose
2. Which subroutines or files are affected (use the graph data above)
3. Risk flags: unclosed file handles, LOOP without EXIT, nested READU locking patterns
4. Your confidence level (High / Medium / Low) and the reason for that confidence

Important:
- Be specific about subroutine names and file handles when mentioned in context
- If the code or graph context doesn't contain enough information, say so clearly
- Do not suggest, imply, or hint at code changes — analysis and explanation only""",
)


class MVAnalysisEngine:
    """
    Main analysis engine. Initialise once; call analyse() for each question.

    Args:
        chroma_path: path to persisted ChromaDB vector store (from setup.py)
        graph_path:  path to graph.json (from setup.py)
        llm_model:   Ollama model name (default: deepseek-coder:13b)
        top_k:       number of RAG chunks to retrieve per query (default: 5)
    """

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        graph_path: str = "./graph.json",
        llm_model: str = "deepseek-coder:13b",
        top_k: int = 5,
    ):
        print(f"  Loading LLM: {llm_model} ...")
        self.llm = Ollama(model=llm_model, temperature=0)

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

    def analyse(self, question: str, subroutine_name: str = None) -> dict:
        """
        Run a full analysis query.

        Args:
            question:        natural-language question from the developer
            subroutine_name: optional subroutine name for graph traversal

        Returns dict with keys:
            answer   - LLM plain-English analysis
            sources  - list of source file paths retrieved from RAG
            impact   - dependency graph data (empty dict if no subroutine given)
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

        # Step 3: Build prompt and call local LLM
        prompt = ANALYSIS_PROMPT.format(
            context=context,
            graph_context=graph_context,
            question=question,
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response,
            "sources": [d.metadata.get("source", "") for d in relevant_docs],
            "impact": impact,
        }
