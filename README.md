# Wiki Chatbot

A Streamlit app that answers user questions with a Wikipedia-backed retrieval and summarization pipeline.

## What it does

- Classifies each prompt as either chit-chat or knowledge-seeking.
- Uses sentence embeddings and a FAISS index to retrieve relevant Wikipedia text.
- Summarizes the retrieved context into a natural-language answer.
- Keeps a lightweight chat history inside the app.

## Project structure

- `app.py` - Streamlit UI and request routing.
- `scraper.py` - Wikipedia documents scraper.
- `classifier.py` - Intent classifier for chit-chat vs. knowledge questions.
- `retriever.py` - Document embedding and FAISS retrieval.
- `summarizer.py` - Answer generation and simple conversational replies.
- `data/` - Cached documents, embeddings, and FAISS index files.
- `models/` - Saved classifier and vectorizer artifacts.

## Setup

Setup virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Scrape wikipedia documents, build retriever and classifier:

```bash
python scraper.py
python retriever.py
python classifier.py
```

## Run the app

Start the Streamlit interface with:

```bash
streamlit run app.py
```

If the FAISS index is missing, the app will show an error and ask you to run `python retriever.py` first.

## How it works

1. The classifier decides whether the prompt is casual conversation or a factual question.
2. Knowledge questions are embedded and compared against the FAISS index.
3. The most relevant Wikipedia passages are passed into a summarization model.
4. The generated answer is shown in the chat UI, with source text available in an expander.

## Notes

- The app includes a small macOS OpenMP workaround in `app.py` for FAISS and PyTorch.
- The retrieval and model files are generated locally, so the first setup may take a few minutes.
