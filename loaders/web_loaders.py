# loaders/web_loaders.py
import os
import json
import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document

os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# Load URLs from config file instead of hardcoding
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "urls.json")

def load_urls():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f).get("urls", [])
    except Exception as e:
        print(f"Could not load urls.json: {e}")
        return []


def load_web_documents():
    documents = []
    urls = load_urls()
    strainer = SoupStrainer(["article", "main", "section", "p"])

    if not urls:
        print("No URLs found in configs/urls.json")
        return documents

    for url in urls:
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