import os

# Fix macOS OpenMP thread safety issue with FAISS + PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
from classifier import load_classifier, classify
from retriever import load_retriever, retrieve
from generator import load_generator, generate_answer, chit_chat_response

st.set_page_config(page_title="Wiki Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Wikipedia Chatbot")

DISTANCE_THRESHOLD = 0.85

# Load all models once and cache them
@st.cache_resource
def load_all_models():
    clf, embed_model = load_classifier()
    retrieval_model, index, texts = load_retriever()
    generator = load_generator()
    return clf, embed_model, retrieval_model, index, texts, generator

# Check index exists before loading
if not os.path.exists("data/faiss_index.bin"):
    st.error("⚠️ FAISS index not found. Run `python retriever.py` first to build it.")
    st.stop()

with st.spinner("Loading models..."):
    clf, embed_model, retrieval_model, index, texts, generator = load_all_models()

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
            intent = classify(prompt, clf, embed_model)

            if intent == "chit-chat":
                response = chit_chat_response(prompt)
            else:
                docs, distances = retrieve(prompt, retrieval_model, index, texts, top_k=5)
                if min(distances) > DISTANCE_THRESHOLD:
                    response = "I don't have enough information on that in my knowledge base. Try rephrasing, or ask about a different topic."
                else:
                    response = generate_answer(prompt, docs, generator)

        st.markdown(response)

        # Show retrieved docs in expander for knowledge queries
        if intent == "knowledge" and min(distances) <= DISTANCE_THRESHOLD:
            with st.expander("📄 Source documents used"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {doc[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})