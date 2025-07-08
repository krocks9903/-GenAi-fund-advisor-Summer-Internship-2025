import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import fitz  # PyMuPDF
import nltk
from openai import RateLimitError

# --- NLTK Sentence Tokenizer Setup ---
NLTK_PATH = os.path.expanduser(r'C:\Users\krock\nltk_data')
if NLTK_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_PATH)
nltk.download('punkt', download_dir=NLTK_PATH, quiet=True)

def extract_pdf_text(pdf_path):
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

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

pdf_paths = [
    "Data/VBTLX.pdf",
    "Data/AGG.pdf",
    "Data/VFIAX.pdf",
    "Data/PRBLX.pdf", 
    "Data/VCIT.pdf",
    "Data/VTMFX.pdf",
    "Data/VWELX.pdf",
    "Data/VBAIX.pdf"
]

docs = []
skipped = 0

# --- Load and Process JSON files ---
print("Loading and processing JSON files...")
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

# --- Load and Process PDFs ---
print("Loading and processing PDFs...")
for pdf_path in tqdm(pdf_paths):
    try:
        pdf_text = extract_pdf_text(pdf_path)
        pdf_text = clean_text(pdf_text)
        for chunk in chunk_text(pdf_text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": os.path.basename(pdf_path)}
            ))
        print(f"Processed {pdf_path}: {len(docs)} chunks so far")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        skipped += 1

print(f"Loaded {len(docs)} chunks from {len(file_paths)} JSONs and {len(pdf_paths)} PDFs (skipped {skipped}).")

# --- Embedding and Indexing (in batches with retry) ---
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

BATCH_SIZE = 50
total_docs = len(docs)
all_chunks = [docs[i:i+BATCH_SIZE] for i in range(0, total_docs, BATCH_SIZE)]

faiss_vectorstores = []
for i, chunk in enumerate(all_chunks):
    print(f"Processing batch {i+1}/{len(all_chunks)}...")
    while True:
        try:
            vs = FAISS.from_documents(chunk, embedding)
            faiss_vectorstores.append(vs)
            break  # Success
        except RateLimitError:
            print("Rate limit hit! Waiting 60 seconds before retrying...")
            time.sleep(60)

# Merge all the partial indexes
if len(faiss_vectorstores) == 1:
    vectorstore = faiss_vectorstores[0]
else:
    vectorstore = faiss_vectorstores[0]
    for vs in faiss_vectorstores[1:]:
        vectorstore.merge_from(vs)

vectorstore.save_local("faiss_index_fund_data")
t1 = time.time()

print(f"Indexed {len(docs)} documents across {len(file_paths)} JSONs and {len(pdf_paths)} PDFs (skipped {skipped} files).")
print(f"FAISS index saved to 'faiss_index_fund_data'. Took {t1-t0:.1f} seconds.")
