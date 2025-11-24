from typing import Callable, List

try:
    from langchain_openai import OpenAI
except Exception:
    try:
        from langchain.llms import OpenAI
    except Exception:
        OpenAI = None

# RetrievalQA optional
try:
    from langchain.chains import RetrievalQA
except Exception:
    RetrievalQA = None

def summarize_docs(docs: List, model_name: str = "gpt-4o-mini", progress_callback: Callable = None) -> str:
    texts = [getattr(d, 'page_content', str(d)) for d in docs]
    MAX_CHUNKS = 8
    selected = texts[:MAX_CHUNKS]
    prompt = """You are an assistant that summarizes documents concisely.
Provide a short, clear summary (4-8 sentences) of the combined content below. If there are multiple sections, briefly mention key points.

---
{content}
""".format(content="\n\n---\n\n".join(selected))

    if progress_callback:
        progress_callback(0.1, "Preparing prompt for LLM...")

    if OpenAI is None:
        # fallback heuristic summary
        preview = "\n\n---\n\n".join(selected[:2])
        return "SUMMARY (heuristic):\n" + (preview[:1000] + ("..." if len(preview) > 1000 else ""))
    llm = OpenAI(temperature=0, model_name=model_name)
    if progress_callback:
        progress_callback(0.4, "Calling LLM to generate summary...")
    summary = llm(prompt)
    if progress_callback:
        progress_callback(1.0, "Summarization complete.")
    return summary

def build_qa_chain(vectorstore, model_name: str = "gpt-4o-mini", progress_callback: Callable = None):
    if RetrievalQA is not None:
        try:
            llm = OpenAI(temperature=0, model_name=model_name) if OpenAI is not None else None
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            if llm is None:
                def retriever_only(query: str):
                    return retriever.get_relevant_documents(query)
                return retriever_only
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            return qa
        except Exception:
            pass

    def qa_fallback(query: str) -> str:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join([getattr(d, 'page_content', '') for d in docs[:6]])
        if OpenAI is None:
            return "ANSWER (retrieval only):\n" + context[:1000] + ("..." if len(context) > 1000 else "")
        llm = OpenAI(temperature=0, model_name=model_name)
        prompt = f"Answer the user's question based only on the context below. If the answer is not contained, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        return llm(prompt)

    return qa_fallback

def retrieve_documents_with_scores(vectorstore, query: str, k: int = 4):
    if hasattr(vectorstore, "similarity_search_with_score"):
        results = vectorstore.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            out.append({"document": doc, "score": float(score)})
        return out
    else:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return [{"document": d, "score": 0.0} for d in docs]
