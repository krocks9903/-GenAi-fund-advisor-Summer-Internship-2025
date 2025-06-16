import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Define the list of JSON files in your Data folder
file_paths = [
    "Data/fund_risk_metrics.json",
    "Data/fund_metadata.json",
    "Data/Definitions.json"
]

docs = []

for path in file_paths:
    with open(path, "r") as f:
        data = json.load(f)

    # Determine document format based on structure
    if isinstance(data, dict):
        # If JSON is a dictionary (like ticker -> metrics)
        for key, content in data.items():
            doc_text = f"{key}\n{json.dumps(content, indent=2)}"
            docs.append(Document(page_content=doc_text, metadata={"source": os.path.basename(path)}))
    elif isinstance(data, list):
        # If JSON is a list (e.g., definitions or general entries)
        for entry in data:
            doc_text = json.dumps(entry, indent=2)
            docs.append(Document(page_content=doc_text, metadata={"source": os.path.basename(path)}))
    else:
        # Skip unsupported formats
        print(f"Skipped: {path} (unsupported structure)")

# Initialize embeddings
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    openai_api_type="azure",
    openai_api_version=os.getenv("AZURE_EMBED_VERSION"),
)

# Build and save FAISS index
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("faiss_index_fund_data")

print(f"Indexed {len(docs)} documents across {len(file_paths)} files. Saved to 'faiss_index_fund_data'.")
