# PDF Q&A + Summarizer — Project Report

**Title:** PDF Q&A + Summarizer — Streamlit + LangChain + OpenAI

**Objective:** Build an AI-powered tool that ingests PDF documents and enables summarization and question-answering over the document content.

**Tools & Frameworks:**
- Python
- Streamlit (UI)
- LangChain (pipelines + chains)
- OpenAI (embeddings & LLM)
- FAISS (vector store)
- pypdf / LangChain PDF loader

**Approach / Workflow Summary:**
1. Ingest PDF with `PyPDFLoader` (preserves page metadata).
2. Split text into chunks using `RecursiveCharacterTextSplitter`.
3. Create embeddings using `OpenAIEmbeddings`.
4. Store vectors in FAISS and persist the index to disk.
5. Build a RetrievalQA chain for question answering and a map-reduce summarization chain.

**Key Implementation Steps:**
- `ingest.py` implements ingestion and persistence.
- `utils.py` provides summarization and QA helper functions.
- `app.py` is the Streamlit interface with progress bars and citation display.

**Results / Observations:**
- The pipeline is effective for medium-length PDFs (10-100 pages).
- Citation snippets significantly help reduce hallucination by grounding answers.
- Persisting the FAISS index saves time when re-using the same documents.

**Learnings / Future Improvements:**
- Add explicit page-number citations and highlight exact passages in the UI.
- Add model selection and cost-control options (batch embeddings, cheaper models).
- Add automated tests and evaluation with a small QA dataset.\n