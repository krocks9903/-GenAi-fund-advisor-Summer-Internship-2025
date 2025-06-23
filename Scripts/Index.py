import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# --- Robust NLTK sentence tokenizer setup ---
import nltk
# Ensure nltk downloads go to a user-writable path (fixes some Windows issues)
NLTK_PATH = os.path.expanduser(r'C:\Users\krock\nltk_data')
if NLTK_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_PATH)
nltk.download('punkt', download_dir=NLTK_PATH, quiet=True)
nltk.download('punkt_tab', download_dir=NLTK_PATH, quiet=True)

def clean_text(text):
    """Remove problematic characters, collapse whitespace, etc."""
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_chunk=800):
    """Chunk text using NLTK's sentence tokenizer."""
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < max_chunk:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

# --- Main Logic ---
load_dotenv()

file_paths = [
    "Data/fund_risk_metrics.json",
    "Data/fund_metadata.json",
    "Data/Definitions.json"
]

docs = []
skipped = 0

print("Loading and processing files...")
for path in tqdm(file_paths):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            skipped += 1
            continue

    if isinstance(data, dict):
        for key, content in data.items():
            # Only chunk narrative docs, not fund metrics
            if "risk_metrics" in path or "fund_risk_metrics" in path.lower():
                doc_text = clean_text(f"Fund: {key}\nRisk Metrics:\n{json.dumps(content, indent=2)}")
                docs.append(Document(page_content=doc_text, metadata={"source": os.path.basename(path), "key": key}))
            else:
                doc_text = clean_text(f"{key}\n{json.dumps(content, indent=2)}")
                for chunk in chunk_text(doc_text):
                    docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(path), "key": key}))
    elif isinstance(data, list):
        for entry in data:
            doc_text = clean_text(json.dumps(entry, indent=2))
            for chunk in chunk_text(doc_text):
                docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(path)}))
    else:
        print(f"Skipped: {path} (unsupported structure)")
        skipped += 1


# --- Embedding and Indexing ---
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    openai_api_type="azure",
    openai_api_version=os.getenv("AZURE_EMBED_VERSION"),
)

print(f"Creating FAISS index for {len(docs)} documents...")
t0 = time.time()
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("faiss_index_fund_data")
t1 = time.time()

print(f"Indexed {len(docs)} documents across {len(file_paths)} files (skipped {skipped} files).")
print(f"FAISS index saved to 'faiss_index_fund_data'. Took {t1-t0:.1f} seconds.")
