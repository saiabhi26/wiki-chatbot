from transformers import pipeline

# Loads a small but capable summarization model
SUMMARY_MODEL = "facebook/bart-large-cnn"

def load_summarizer():
    return pipeline("summarization", model=SUMMARY_MODEL)

def generate_answer(query, retrieved_docs, summarizer):
    """Combine retrieved docs and summarize into an answer."""
    context = " ".join(retrieved_docs)[:3000]  # Cap tokens
    prompt = f"Question: {query}\n\nContext: {context}"

    result = summarizer(prompt, max_length=200, min_length=40, do_sample=False)
    return result[0]["summary_text"]

# Simple chit-chat responses
CHIT_CHAT_RESPONSES = {
    "hello": "Hey! Ask me anything about AI, history, science, and more!",
    "hi": "Hi there! What would you like to know?",
    "how are you": "I'm doing great! Ready to answer your questions.",
    "bye": "Goodbye! Come back anytime.",
    "thanks": "You're welcome!",
    "who are you": "I'm a Wikipedia-powered chatbot. Ask me anything!",
    "default": "I'm not sure how to respond to that. Try asking me a knowledge question!"
}

def chit_chat_response(query):
    query_lower = query.lower().strip()
    for key in CHIT_CHAT_RESPONSES:
        if key in query_lower:
            return CHIT_CHAT_RESPONSES[key]
    return CHIT_CHAT_RESPONSES["default"]