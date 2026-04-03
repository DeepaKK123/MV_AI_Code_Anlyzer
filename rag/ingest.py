"""
rag/ingest.py
RAG corpus ingestion — accepts ALL files in mv_source regardless of extension.
MV files like ORD.PROCESS have .PROCESS as suffix — accepted by passing extensions=None.
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def load_files_from_dir(directory: str, extensions) -> list:
    """
    Load all files from directory recursively.
    extensions=None  -> accept every file (used for mv_source)
    extensions=[...] -> accept only matching suffixes (used for documents)
    """
    docs = []
    path = Path(directory)
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        # Accept all files if extensions is None, otherwise match suffix
        if extensions is None or f.suffix.lower() in extensions:
            try:
                loader = TextLoader(str(f), encoding="utf-8")
                docs.extend(loader.load())
            except Exception:
                try:
                    loader = TextLoader(str(f), encoding="latin-1")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"  WARNING: Could not load {f}: {e}")
    return docs


def ingest_corpus(
    source_dir: str,
    docs_dir: str,
    chroma_path: str = "./chroma_db"
) -> Chroma:
    """
    Load, chunk, embed, and store all source files and documents.

    source_dir: folder containing MV BASIC files (any name, any extension)
    docs_dir:   folder containing .txt contract/schema/Layout documents
    chroma_path: where ChromaDB persists its data (auto-created)
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docs = []

    # Load ALL files from mv_source — no extension filtering
    source_docs = load_files_from_dir(source_dir, extensions=None)
    if source_docs:
        print(f"  Loaded {len(source_docs)} source files from {source_dir}")
        docs += source_docs
    else:
        print(f"  WARNING: No source files found in {source_dir}")

    # Load .txt documents only
    txt_docs = load_files_from_dir(docs_dir, extensions=[".txt"])
    if txt_docs:
        print(f"  Loaded {len(txt_docs)} documents from {docs_dir}")
        docs += txt_docs
    else:
        print(f"  WARNING: No .txt files found in {docs_dir}")

    if not docs:
        raise RuntimeError(
            "No documents loaded. Add MV BASIC files to mv_source/ "
            "and/or .txt files to documents/ before running setup.py"
        )

    # Chunk at MV BASIC subroutine and insert boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nSUBROUTINE ", "\nFUNCTION ", "\n$INSERT ", "\n* ", "\n!", "\n"],
    )

    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    # Tag each chunk with its source type
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk.metadata["source_type"] = (
            "document" if src.lower().endswith(".txt") else "source_code"
        )

    print(f"  Embedding {len(chunks)} chunks with nomic-embed-text ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    print(f"  ChromaDB persisted to {chroma_path}")
    return vectorstore