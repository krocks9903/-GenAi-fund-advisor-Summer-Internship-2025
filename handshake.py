import requests
import os
from dotenv import load_dotenv
load_dotenv()

chat_api_url = os.getenv('GPT_URL')
chat_api_key = os.getenv('GPT-4.1')

chat_headers = {
    "Content-Type": "application/json",
    "api-key": chat_api_key
}

chat_payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you hear me?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}

chat_response = requests.post(chat_api_url, headers=chat_headers, json=chat_payload)
print("Chat API Handshake Status:", chat_response.status_code)
print("Chat Response:", chat_response.json())

embedding_api_url = os.getenv('Embedding_URL')
embedding_api_key = os.getenv('Embedding')

embedding_headers = {
    "Content-Type": "application/json",
    "api-key": embedding_api_key
}

embedding_payload = {
    "input": "This is a test for embedding.",
}

embedding_response = requests.post(embedding_api_url, headers=embedding_headers, json=embedding_payload)
print("Embedding API Handshake Status:", embedding_response.status_code)
print("Embedding Response:", embedding_response.json())
