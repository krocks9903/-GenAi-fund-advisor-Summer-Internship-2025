import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

# Load FAISS index and set up retriever
vectorstore = FAISS.load_local(
    "faiss_index_fund_data",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# Configure Azure OpenAI chat model
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    model="gpt-4.1",
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_CHAT_VERSION"),
    temperature=0,
)

# Streamlit page configuration
st.set_page_config(page_title="GenAI Fund Advisor", layout="wide")
st.markdown("<h1 style='color:#f97316;'>GenAI Fund Advisor</h1>", unsafe_allow_html=True)
st.markdown("Ask questions about mutual funds, definitions, or risk metrics.")

# Clear conversation button
if st.button("Clear Conversation"):
    for key in ("memory", "qa_chain"):
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


custom_prompt = PromptTemplate.from_template("""
You are a financial assistant helping users with mutual fund questions.

You have access to:
- Definitions of terms (e.g., Sortino, Sharpe)
- Fund risk metrics (Sharpe, Sortino, Treynor, Std Dev)
- Fund metadata (inception date, AUM, expense ratio)

Before answering, use chain-of-thought reasoning:

STEP 1: Identify the question type:
  - Definition/info question? ("What is AUM?")
  - Fund comparison? ("Compare VWELX vs VFIAX")
  - Fund recommendation? ("Suggest a low-risk long-term fund")

STEP 2: Handle accordingly:
  - For definitions: Explain the term from context.
  - For comparisons: Compare metrics from context across funds.
  - For recommendations:
      a. Infer user profile (risk, time horizon, ESG, etc.)
      b. Filter based on context metrics (Sortino, Sharpe, Expense, etc.)
      c. Recommend and justify based on real data.

If context is missing, reply: "That information is not available in the provided context."

Context:
{context}

Question:
{question}
""")

# Initialize memory and QA chain
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        output_key="answer"
    )

# Display chat history
if st.session_state.memory.chat_memory.messages:
    st.markdown("### Chat History")
    for msg in st.session_state.memory.chat_memory.messages:
        role = "You" if msg.type == "human" else "GenAI"
        bubble_class = "user" if msg.type == "human" else "ai"
        st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
            <strong>{role}:</strong><br>{msg.content}
        </div>
        """, unsafe_allow_html=True)

# User query input
query = st.text_input("Your Question", placeholder="E.g. What is Sortino Ratio? Or Compare PRBLX and FSPTX")

if query:
    result = st.session_state.qa_chain.invoke({"question": query})
    answer = result.get("answer")
    source_docs = result.get("source_documents", [])
    sources = "\n".join(doc.metadata.get("source", "N/A") for doc in source_docs)

    if not answer or "context does not include" in answer.lower():
        st.warning("No relevant data found for that question.")
    else:
        st.markdown("### Response:")
        st.markdown(f"<div class='custom-answer'>{answer}</div>", unsafe_allow_html=True)

        if sources.strip():
            st.markdown(f"<p><b>Sources:</b><br>{sources}</p>", unsafe_allow_html=True)

        if st.button("Export Response as PDF"):
            class CustomPDF(FPDF):
                def header(self):
                    self.set_font("Arial", size=12)
                    self.cell(0, 10, "GenAI Fund Advisor Response", ln=True, align="C")

            pdf = CustomPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, answer)
            if sources:
                pdf.ln(5)
                pdf.multi_cell(0, 10, f"Sources:\n{sources}")

            pdf_path = "response.pdf"
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="response.pdf")

# CSS styling
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
            color: black;
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
        .chat-bubble {
            border-radius: 12px;
            padding: 0.75em 1em;
            margin: 0.5em 0;
            max-width: 95%;
        }
        .chat-bubble.user {
            background-color: #fef3c7;
            border-left: 5px solid #f59e0b;
        }
        .chat-bubble.ai {
            background-color: #ecfccb;
            border-left: 5px solid #4ade80;
        }
    </style>
""", unsafe_allow_html=True)
