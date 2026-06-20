from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import pickle
import os
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"

# Small labeled dataset to train the router
TRAINING_DATA = [
    # Chit-chat
    ("hello", "chit-chat"), ("hi there", "chit-chat"), ("hey", "chit-chat"),
    ("how are you", "chit-chat"), ("how are you doing", "chit-chat"),
    ("what's up", "chit-chat"), ("good morning", "chit-chat"), ("good evening", "chit-chat"),
    ("bye", "chit-chat"), ("goodbye", "chit-chat"), ("see you later", "chit-chat"),
    ("thanks", "chit-chat"), ("thank you", "chit-chat"), ("appreciate it", "chit-chat"),
    ("who are you", "chit-chat"), ("what can you do", "chit-chat"), ("what are you", "chit-chat"),
    ("tell me a joke", "chit-chat"), ("make me laugh", "chit-chat"),
    ("you're great", "chit-chat"), ("nice to meet you", "chit-chat"),
    ("lol", "chit-chat"), ("ok cool", "chit-chat"), ("that's interesting", "chit-chat"),
    ("cool", "chit-chat"), ("nice", "chit-chat"), ("haha", "chit-chat"),
    ("are you human", "chit-chat"), ("are you a robot", "chit-chat"),
    ("what's your name", "chit-chat"), ("do you sleep", "chit-chat"),

    # Knowledge - topic-specific
    ("what is machine learning", "knowledge"), ("explain quantum computing", "knowledge"),
    ("how does photosynthesis work", "knowledge"), ("who invented the telephone", "knowledge"),
    ("what caused world war 2", "knowledge"), ("tell me about climate change", "knowledge"),
    ("how does bitcoin work", "knowledge"), ("what is python used for", "knowledge"),
    ("history of the renaissance", "knowledge"), ("what is a black hole", "knowledge"),
    ("how are genes inherited", "knowledge"), ("what is artificial intelligence", "knowledge"),
    ("explain space exploration", "knowledge"), ("what is deep learning", "knowledge"),
    ("what is dna", "knowledge"), ("explain machine learning", "knowledge"),

    # Knowledge - short, math-like, or generically phrased
    # (the kind of query the old classifier consistently misrouted)
    ("what is 1+1", "knowledge"), ("what is 2+2", "knowledge"),
    ("what do you get when you add 1 and 1", "knowledge"),
    ("what is the square root of 9", "knowledge"),
    ("what is the formula for the area of a circle", "knowledge"),
    ("what is the perimeter of a circle", "knowledge"),
    ("what is pythagoras theorem", "knowledge"),
    ("how many planets are in the solar system", "knowledge"),
    ("what is the speed of light", "knowledge"),
    ("how old is the universe", "knowledge"),
    ("what year did world war 2 end", "knowledge"),
    ("who is albert einstein", "knowledge"),
    ("what is gravity", "knowledge"),
]

CLASSIFIER_PATH = "models/classifier.pkl"

def train_classifier():
    os.makedirs("models", exist_ok=True)
    texts, labels = zip(*TRAINING_DATA)
    
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Embedding training examples...")
    X = embed_model.encode(list(texts))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    print("✅ Classifier trained and saved")

def load_classifier():
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
    return clf, embed_model

def classify(query, clf, vectorizer):
    X = embed_model.encode([query])  
    return clf.predict(X)[0]  # "chit-chat" or "knowledge"

if __name__ == "__main__":
    train_classifier()
    clf, embed_model = load_classifier()
    tests = ["hello!", "explain machine learning", "how are you?", "what is DNA?",
        "1+1 = ?", "what do you get when you add the number 1 and 1",
        "what is formula for perimeter of circle?",]
    for t in tests:
        print(f"  '{t}' → {classify(t, clf, embed_model)}")