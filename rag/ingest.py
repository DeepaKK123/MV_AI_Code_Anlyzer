"""
rag/ingest.py
RAG corpus ingestion: loads UniData .B source files and .txt documents,
chunks them at subroutine boundaries, embeds with nomic-embed-text via
Ollama, and persists to ChromaDB on disk.

Run via setup.py. Can also be called directly:
    python -m rag.ingest
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def ingest_corpus(
    source_dir: str,
    docs_dir: str,
    chroma_path: str = "./chroma_db"
) -> Chroma:
    """
    Load, chunk, embed, and store all source files and documents.

    source_dir: folder containing UniData .B source files
    docs_dir:   folder containing .txt contract/schema documents
    chroma_path: where ChromaDB persists its data (auto-created)

    Returns the Chroma vectorstore instance.
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    docs = []

    # Load UniData .B source files
    source_path = Path(source_dir)
    b_files = list(source_path.rglob("*.B")) + list(source_path.rglob("*.bas"))
    if b_files:
        print(f"  Loading {len(b_files)} source files from {source_dir} ...")
        code_loader = DirectoryLoader(
            source_dir,
            glob="**/*.B",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "errors": "ignore"},
            silent_errors=True,
        )
        docs += code_loader.load()
    else:
        print(f"  WARNING: No .B files found in {source_dir}")

    # Load contracts, schema docs (plain text)
    docs_path = Path(docs_dir)
    txt_files = list(docs_path.rglob("*.txt"))
    if txt_files:
        print(f"  Loading {len(txt_files)} documents from {docs_dir} ...")
        doc_loader = DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "errors": "ignore"},
            silent_errors=True,
        )
        docs += doc_loader.load()
    else:
        print(f"  WARNING: No .txt files found in {docs_dir}")

    if not docs:
        raise RuntimeError(
            "No documents loaded. Add .B files to unidata_source/ "
            "and/or .txt files to documents/ before running setup.py"
        )

    # Chunk at UniData BASIC subroutine and insert boundaries
    # Separators tuned for MV BASIC structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nSUBROUTINE ", "\nFUNCTION ", "\n$INSERT ", "\n* ", "\n!", "\n"],
    )

    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    # Tag each chunk with its source type for filtering later
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk.metadata["source_type"] = (
            "source_code" if src.upper().endswith(".B") or src.lower().endswith(".bas")
            else "document"
        )

    print(f"  Embedding {len(chunks)} chunks with nomic-embed-text (this may take a few minutes) ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    print(f"  ChromaDB persisted to {chroma_path}")
    return vectorstore
