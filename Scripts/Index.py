import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define JSON file paths
base_path = "Data"
file_paths = [
    os.path.join(base_path, "Definitions.json"),
    os.path.join(base_path, "fund_metadata.json"),
    os.path.join(base_path, "fund_risk_metrics.json"),
]

# Load all JSON files into a single list of documents
docs = []
for path in file_paths:
    loader = JSONLoader(file_path=path, jq_schema=".", text_content=False)
    docs.extend(loader.load())

# Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Initialize Azure OpenAI Embeddings
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    openai_api_type="azure",
    openai_api_version=os.getenv("AZURE_EMBED_VERSION"),
)

# Create FAISS index from the document chunks
vectorstore = FAISS.from_documents(chunks, embedding)

# Save the FAISS index locally
vectorstore.save_local("faiss_index_fund_data")

print("Indexing complete. FAISS database saved to 'faiss_index_fund_data'.")
# This script loads JSON files, splits them into chunks, creates a FAISS index using Azure OpenAI embeddings,