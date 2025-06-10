import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize embeddings
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    api_version=os.getenv("AZURE_EMBED_VERSION"),
    chunk_size=1000,
)

# Load FAISS and retriever
vectorstore = FAISS.load_local("faiss_index_fund_data", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Azure OpenAI Chat LLM
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    model="gpt-4.1",
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_CHAT_VERSION"),
    temperature=0,
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="GenAI Fund Advisor", layout="wide")
st.markdown("<h1 style='color:#f97316;'>GenAI Fund Advisor</h1>", unsafe_allow_html=True)
st.markdown("Ask questions about mutual funds, definitions, or risk metrics.")

# User query
query = st.text_input("Your Question", placeholder="E.g. What is a fund with a low expense ratio?")

if query:
    result = qa_chain.invoke({"query": query})
    answer = result.get("result") if isinstance(result, dict) else result

    if "provided context does not include" in answer.lower():
        st.warning("No relevant data found for that question.")
    else:
        st.markdown("### Response:")
        st.markdown(f"<div class='custom-answer'>{answer}</div>", unsafe_allow_html=True)

        # PDF Export with UTF-8 support
        if st.button("Export Response as PDF"):
            class CustomPDF(FPDF):
                def header(self):
                    self.set_font("DejaVu", size=12)
                    self.cell(0, 10, "GenAI Fund Advisor Response", ln=True, align="C")

            pdf = CustomPDF()
            pdf.add_page()

            # Register TTF font with UTF-8 support
            font_path = "DejaVuSans.ttf"  # Ensure this file is in your directory
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=12)

            for line in answer.split("\n"):
                pdf.multi_cell(0, 10, line)

            pdf_path = "answer_output.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="response.pdf")

# Summer theme styling
st.markdown("""
    <style>
        .stApp {
            background-color: #fffbea;
            color: #1f2937;
            font-family: "Segoe UI", sans-serif;
        }
        h1 {
            margin-bottom: 0.2em;
        }
        .stTextInput > div > div > input {
            font-weight: 500;
            background-color: #fffdf0;
            border: 2px solid #facc15;
            border-radius: 8px;
            padding: 0.6em;
            color: black; /*  Set input text color to black */
        }
        .stButton > button {
            background-color: #f97316;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #fb923c;
        }
        .custom-answer {
            background-color: #fef9c3;
            color: #111827;
            padding: 1em;
            border-radius: 10px;
            border-left: 6px solid #22c55e;
            font-size: 1.05em;
        }
    </style>
""", unsafe_allow_html=True)
