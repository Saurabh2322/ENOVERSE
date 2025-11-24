
import os
import re
from typing import Callable, List
import fitz  # PyMuPDF

def safe_get_text_splitter(chunk_size=1500, chunk_overlap=120):
    """
    Tries modern langchain_text_splitters.
    Falls back to simple Python paragraph splitter if anything fails.
    """

    # Try modern package (may fail if torch/transformers unavailable)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        pass

    # Secondary fallback (rarely used)
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        pass

    # FINAL SAFE FALLBACK — paragraph splitter
    class SimpleSplitter:
        """Simple paragraph-based splitter."""
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, docs):
            out = []
            for d in docs:
                text = getattr(d, "page_content", str(d))
                paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

                cur = ""
                for p in paras:
                    if len(cur) + len(p) + 2 <= self.chunk_size:
                        cur = (cur + "\n\n" + p).strip() if cur else p
                    else:
                        out.append(type(d)(page_content=cur, metadata=getattr(d, "metadata", {})))
                        cur = p

                if cur:
                    out.append(type(d)(page_content=cur, metadata=getattr(d, "metadata", {})))

            return out

    return SimpleSplitter(chunk_size, chunk_overlap)

def _load_pdf_with_pymupdf(pdf_path: str):
    """Extract pages using PyMuPDF into LangChain Document objects."""

    try:
        from langchain.schema import Document
    except Exception:
        class Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
    return pages

class _STSimpleWrapper:
    """
    Minimal wrapper around sentence-transformers that avoids torch import at module load.
    Loads model ONLY when used.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ImportError(
                "sentence-transformers must be installed: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [list(map(float, x)) for x in self.model.encode(texts, show_progress_bar=False)]

    def embed_query(self, text):
        return list(map(float, self.model.encode([text])[0]))

try:
    from langchain.vectorstores import Chroma
    _CHROMA_OK = True
except Exception:
    _CHROMA_OK = False

def ingest_pdf_to_faiss(
    pdf_path: str,
    persist_path: str = None,
    progress_callback: Callable = None,
    hf_model_name: str = "all-MiniLM-L6-v2"
):
    """
    Ingest PDF → split → embed → store → return vectorstore
    """

    # Load PDF
    if progress_callback: progress_callback(0.05, "Loading PDF...")
    docs = _load_pdf_with_pymupdf(pdf_path)

    # Split
    if progress_callback: progress_callback(0.15, "Splitting text...")
    splitter = safe_get_text_splitter(chunk_size=1500, chunk_overlap=120)
    splitted_docs = splitter.split_documents(docs)

    # Embeddings
    if progress_callback: progress_callback(0.30, "Embedding chunks locally...")
    embeddings = _STSimpleWrapper(hf_model_name)

    # Storage directory
    if persist_path is None:
        persist_path = "chroma_index"

    os.makedirs(persist_path, exist_ok=True)

    # Chroma check
    if not _CHROMA_OK:
        raise RuntimeError("Chroma not installed. Install via: pip install chromadb")

    # Vectorstore
    if progress_callback: progress_callback(0.70, "Creating Chroma vectorstore...")
    vectorstore = Chroma.from_documents(
        splitted_docs,
        embeddings,
        persist_directory=persist_path
    )

    if progress_callback: progress_callback(1.0, "Done ingesting.")

    return vectorstore

def load_vectorstore(persist_path: str, hf_model_name="all-MiniLM-L6-v2"):
    """Load saved Chroma index."""
    if not os.path.isdir(persist_path):
        raise ValueError(f"Directory not found: {persist_path}")

    if not _CHROMA_OK:
        raise RuntimeError("Chroma is not installed. pip install chromadb")

    embeddings = _STSimpleWrapper(hf_model_name)

    return Chroma(persist_directory=persist_path, embedding_function=embeddings)