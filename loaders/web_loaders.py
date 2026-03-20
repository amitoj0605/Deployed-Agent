# loaders/web_loaders.py
import os
import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document

os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

URLS = [
    "https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai",
    "https://www.cognigy.com/agentic-ai/generative-ai-vs-agentic-ai",
    "https://www.geeksforgeeks.org/artificial-intelligence/gen-ai-vs-ai-agents-vs-agentic-ai/",
]

# Removed Adobe URL — blocks scrapers consistently


def load_web_documents():
    documents = []
    strainer = SoupStrainer(["article", "main", "section", "p"])

    for url in URLS:
        print(f"Loading: {url}")
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": os.environ["USER_AGENT"]
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser", parse_only=strainer)
            text = " ".join(soup.get_text().split())

            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": url}
                ))
                print(f"  ✓ Loaded {len(text)} chars")
            else:
                print(f"  ✗ Empty content, skipping")

        except Exception as e:
            print(f"  ✗ Failed to load {url}: {e}")
            continue

    return documents