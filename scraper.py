import wikipediaapi
import json
import os

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='wiki-chatbot/1.0'
)

TOPICS = [
    "Artificial intelligence", "Machine learning", "Python (programming language)",
    "Climate change", "Space exploration", "World War II",
    "Quantum computing", "Genetics", "Cryptocurrency", "Renaissance", "Mathematics"
]

def scrape_topic(topic, max_articles=50):
    """Scrape a topic's main page and linked pages."""
    documents = []
    page = wiki.page(topic)
    if not page.exists():
        return documents

    # Add the main page sections
    for section in page.sections:
        if section.text.strip():
            documents.append({
                "title": page.title,
                "section": section.title,
                "text": section.text.strip(),
                "topic": topic
            })

    # Follow links to get more articles
    count = 0
    for link_title in page.links:
        if count >= max_articles:
            break
        linked_page = wiki.page(link_title)
        if linked_page.exists() and linked_page.text:
            for section in linked_page.sections:
                if section.text.strip():
                    documents.append({
                        "title": linked_page.title,
                        "section": section.title,
                        "text": section.text.strip(),
                        "topic": topic
                    })
        count += 1
    return documents

def run_scraper():
    all_docs = []
    for topic in TOPICS:
        print(f"Scraping: {topic}...")
        docs = scrape_topic(topic)
        all_docs.extend(docs)
        print(f"  → {len(docs)} sections collected")

    os.makedirs("data", exist_ok=True)
    with open("data/documents.json", "w") as f:
        json.dump(all_docs, f, indent=2)
    print(f"\n✅ Total documents saved: {len(all_docs)}")

if __name__ == "__main__":
    run_scraper()