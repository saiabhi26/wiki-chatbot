import os

# Fix macOS OpenMP thread safety issue with FAISS + PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
from classifier import load_classifier, classify
from retriever import load_retriever, retrieve
from summarizer import load_summarizer, generate_answer, chit_chat_response

st.set_page_config(page_title="Wiki Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Wikipedia Chatbot")

# Load all models once and cache them
@st.cache_resource
def load_all_models():
    clf, vectorizer = load_classifier()
    retrieval_model, index, texts = load_retriever()
    summarizer = load_summarizer()
    return clf, vectorizer, retrieval_model, index, texts, summarizer

# Check index exists before loading
if not os.path.exists("data/faiss_index.bin"):
    st.error("⚠️ FAISS index not found. Run `python retriever.py` first to build it.")
    st.stop()

with st.spinner("Loading models..."):
    clf, vectorizer, retrieval_model, index, texts, summarizer = load_all_models()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = classify(prompt, clf, vectorizer)

            if intent == "chit-chat":
                response = chit_chat_response(prompt)
            else:
                docs = retrieve(prompt, retrieval_model, index, texts, top_k=5)
                response = generate_answer(prompt, docs, summarizer)

        st.markdown(response)

        # Show retrieved docs in expander for knowledge queries
        if intent == "knowledge":
            with st.expander("📄 Source documents used"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {doc[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})