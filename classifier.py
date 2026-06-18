from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Small labeled dataset to train the router
TRAINING_DATA = [
    # Chit-chat
    ("hello", "chit-chat"), ("hi there", "chit-chat"), ("how are you", "chit-chat"),
    ("what's up", "chit-chat"), ("good morning", "chit-chat"), ("bye", "chit-chat"),
    ("thanks", "chit-chat"), ("who are you", "chit-chat"), ("what can you do", "chit-chat"),
    ("tell me a joke", "chit-chat"), ("you're great", "chit-chat"), ("nice to meet you", "chit-chat"),
    ("lol", "chit-chat"), ("ok cool", "chit-chat"), ("that's interesting", "chit-chat"),

    # Knowledge
    ("what is machine learning", "knowledge"), ("explain quantum computing", "knowledge"),
    ("how does photosynthesis work", "knowledge"), ("who invented the telephone", "knowledge"),
    ("what caused world war 2", "knowledge"), ("tell me about climate change", "knowledge"),
    ("how does bitcoin work", "knowledge"), ("what is python used for", "knowledge"),
    ("history of the renaissance", "knowledge"), ("what is a black hole", "knowledge"),
    ("how are genes inherited", "knowledge"), ("what is artificial intelligence", "knowledge"),
    ("explain space exploration", "knowledge"), ("what is deep learning", "knowledge"),
]

CLASSIFIER_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def train_classifier():
    os.makedirs("models", exist_ok=True)
    texts, labels = zip(*TRAINING_DATA)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression()
    clf.fit(X, labels)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print("✅ Classifier trained and saved")

def load_classifier():
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer

def classify(query, clf, vectorizer):
    X = vectorizer.transform([query])
    return clf.predict(X)[0]  # "chit-chat" or "knowledge"

if __name__ == "__main__":
    train_classifier()
    clf, vec = load_classifier()
    tests = ["hello!", "explain machine learning", "how are you?", "what is DNA?"]
    for t in tests:
        print(f"  '{t}' → {classify(t, clf, vec)}")