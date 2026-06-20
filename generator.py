from transformers import T5Tokenizer, T5ForConditionalGeneration

ANSWER_MODEL = "google/flan-t5-base"

def load_generator():
    tokenizer = T5Tokenizer.from_pretrained(ANSWER_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(ANSWER_MODEL)
    return tokenizer, model

def generate_answer(query, retrieved_docs, summarizer):
    """Combine retrieved docs and generate a focused answer to the query."""
    tokenizer, model = summarizer
    context = " ".join(retrieved_docs)[:3000]
    prompt = (
        f"Answer the question using only the context below. "
        f"If the context doesn't contain the answer, say you don't have enough information.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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