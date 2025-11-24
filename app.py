import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os, tempfile

from ingest import ingest_pdf_to_faiss, load_vectorstore
from utils import summarize_docs, build_qa_chain, retrieve_documents_with_scores

st.set_page_config(page_title="PDF Q&A + Summarizer", layout="wide")
st.title("PDF Q&A + Summarizer — PyMuPDF + Chroma + Local HF Embeddings")

st.markdown(
    "Upload a PDF, ingest it to create a persistent Chroma vectorstore (local HuggingFace embeddings),"
    " then ask questions or get a summary. This version uses PyMuPDF and runs embeddings locally (no OpenAI calls required)."
)

col1, col2 = st.columns([3,1])

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.success(f"Saved uploaded PDF to {tmp_path}")
    else:
        st.info("No upload detected. For quick demo you can use the sample PDF bundled with this environment.")
        if st.button("Use demo PDF"):
            tmp_path = r"/mnt/data/EONVERSE AI Intern – Screening Challenge.pdf"
            st.success(f"Using demo PDF: {tmp_path}")

with col2:
    persist_dir = st.text_input("Persistence directory (local)", value="chroma_index")
    hf_model = st.text_input("HF model (sentence-transformers)", value="all-MiniLM-L6-v2")
    ingest_button = st.button("Ingest PDF (create/update persistent index)")

if 'tmp_path' in locals() and ingest_button:
    progress = st.progress(0)
    status = st.empty()
    status.text("Starting ingestion...")
    try:
        vs = ingest_pdf_to_faiss(
            tmp_path,
            persist_path=persist_dir,
            progress_callback=lambda p, m: (progress.progress(int(p*100)), status.text(m)),
            hf_model_name=hf_model
        )
        st.success("Ingestion complete and saved to disk.")
        st.session_state['vectorstore_path'] = persist_dir
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

# Load existing index
if 'vectorstore_path' in st.session_state:
    if st.button("Load persistent index from disk"):
        try:
            with st.spinner("Loading vectorstore from disk..."):
                vs = load_vectorstore(st.session_state['vectorstore_path'], hf_model_name=hf_model)
                st.session_state['vectorstore'] = vs
            st.success("Vectorstore loaded.")
        except Exception as e:
            st.error(f"Failed to load vectorstore: {e}")

if 'vectorstore' in st.session_state:
    vectorstore = st.session_state['vectorstore']
    st.header("Summarize document")
    if st.button("Generate Summary"):
        progress = st.progress(0)
        status = st.empty()
        status.text("Preparing retrieval for summarization...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})
        try:
            docs = retriever.get_relevant_documents("") if hasattr(retriever, "get_relevant_documents") else []
        except Exception:
            docs = []
        if len(docs) == 0:
            st.info("To generate a full-document summary, re-upload the PDF and click Ingest.")
        else:
            status.text("Generating summary from retrieved chunks...")
            summary = summarize_docs(docs, progress_callback=lambda p, m: (progress.progress(int(p*100)), status.text(m)))
            st.subheader("Summary")
            st.write(summary)

    st.header("Ask questions")
    question = st.text_input("Enter your question about the document:")
    if question:
        st.write("Searching and generating answer (with citations)...")
        progress_bar = st.progress(0)
        qa = build_qa_chain(vectorstore, progress_callback=lambda p, m: progress_bar.progress(int(p*100)))
        docs_with_scores = retrieve_documents_with_scores(vectorstore, question, k=4)
        with st.spinner("Generating answer..."):
            answer = qa.run(question) if hasattr(qa, "run") else qa(question)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Citations / source snippets")
        for i, item in enumerate(docs_with_scores):
            doc = item['document']
            score = item['score']
            meta = getattr(doc, "metadata", {})
            page = meta.get("page", "unknown")
            source = meta.get("source", meta.get("file_path", "unknown"))
            st.markdown(f"**Source {i+1}** — page: {page}, score: {score:.4f}, source: {source}")
            snippet = doc.page_content[:500].strip().replace('\n', ' ')
            st.write(snippet)
            st.markdown("---")

    st.markdown("---")
    if st.button("Clear Session and Unload Vectorstore"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Session cleared. Upload a new file.")
else:
    st.info("No vectorstore available. Upload and ingest a PDF to get started.")
