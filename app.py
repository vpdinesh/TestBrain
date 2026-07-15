#!/usr/bin/env python3
"""
TestBrain — AI-Powered Test Intelligence Assistant
====================================================
A Streamlit web app that answers questions about your test automation
project using RAG (Retrieval-Augmented Generation).

Runs 100% locally — no API keys, no internet needed.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
OLLAMA_MODEL = os.environ.get("TESTBRAIN_MODEL", "llama3.2:1b")


# ── RAG Setup ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_rag_chain():
    """Load the RAG chain (cached for performance)."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)

    prompt_template = PromptTemplate(
        template="""You are TestBrain, an AI-powered test intelligence assistant.
Use the following context from test reports, documentation, and results to answer the question accurately.
If you don't know the answer from the context, say "I don't have that information in the indexed documents."
Always cite which source document the information comes from.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    return qa_chain, vectorstore


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="TestBrain",
        page_icon="🧠",
        layout="wide"
    )

    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); padding: 24px; border-radius: 12px; margin-bottom: 20px;">
        <h1 style="color: #f59e0b; margin: 0;">🧠 TestBrain</h1>
        <p style="color: #a3a3a3; margin: 5px 0 0;">AI-Powered Test Intelligence — Ask questions about your test automation project</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### How It Works")
        st.markdown("""
        1. **Index** — Your test reports & docs are chunked and embedded
        2. **Search** — Your question is matched to relevant chunks
        3. **Generate** — Local LLM answers using the retrieved context
        """)
        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("""
        - 🧠 **LLM:** Ollama + Llama 3.2
        - 🔍 **Embeddings:** Sentence-BERT
        - 📦 **Vector DB:** ChromaDB
        - 🔗 **Framework:** LangChain
        - 🎨 **UI:** Streamlit
        """)
        st.markdown("---")
        st.markdown("### Sample Questions")
        sample_questions = [
            "What test suites are automated?",
            "What failure codes are tracked?",
            "How many test cases exist?",
            "What environments are supported?",
            "What security tests are implemented?",
            "How does failure tracking work?",
            "What databases are used?",
            "What is the test coverage?",
        ]
        for q in sample_questions:
            if st.button(q, key=q):
                st.session_state["question"] = q

    # Check if DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        st.error("⚠️ Vector database not found! Index your documents first:")
        st.code("python3 indexer.py --path /path/to/your/test/repo", language="bash")
        st.markdown("""
        **Steps:**
        1. Point `--path` to your test automation repository
        2. The indexer will scan for `.md` and `.json` files
        3. Then come back here and refresh
        """)
        return

    # Load RAG chain
    with st.spinner("Loading TestBrain..."):
        qa_chain, vectorstore = load_rag_chain()

    chunk_count = vectorstore._collection.count()
    st.success(f"🧠 TestBrain ready — {chunk_count} knowledge chunks indexed")

    # Query input
    question = st.text_input(
        "Ask about your test automation project:",
        value=st.session_state.get("question", ""),
        placeholder="e.g., What failure codes are tracked in the system?"
    )

    if question:
        with st.spinner("🔍 Searching & generating answer..."):
            result = qa_chain.invoke({"query": question})

        # Display answer
        st.markdown("### 💡 Answer")
        st.markdown(result["result"])

        # Display sources
        if result.get("source_documents"):
            st.markdown("---")
            st.markdown("### 📄 Sources")
            for i, doc in enumerate(result["source_documents"][:3]):
                source = doc.metadata.get("source", "unknown")
                with st.expander(f"Source {i+1}: {source}"):
                    st.text(doc.page_content[:500])


if __name__ == "__main__":
    main()
