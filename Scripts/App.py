import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
from fpdf import FPDF

# --- Load environment variables
load_dotenv()

# --- Initialize embeddings
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    api_version=os.getenv("AZURE_EMBED_VERSION"),
    chunk_size=1000,
)

# --- Load FAISS index and set up retriever
vectorstore = FAISS.load_local(
    "faiss_index_fund_data",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# --- Configure Azure OpenAI chat model
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    model="gpt-4.1",
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_CHAT_VERSION"),
    temperature=0,
)

custom_prompt = PromptTemplate.from_template("""
INSTRUCTION: First, provide a direct and concise answer to the user's question in 2-3 sentences and label it as 'Answer:'. After that, clearly label and provide your full explanation or reasoning as 'Explanation:' based on the chain-of-thought steps.

You are an expert mutual fund assistant. For any question about a specific ticker or fund, use this rigorous chain-of-thought reasoning:

Step 1: Identify user intent and which ticker(s) or fund(s) are referenced.
Step 2: Retrieve and present the most relevant sections from each fund's prospectus summary:
- For risk questions: Use "Principal Risks" and risk disclosures.
- For investment approach: Use "Investment Objective" and "Strategy".
- For cost questions: Use "Fees and Expenses".
- For suitability: Use "Who Should Invest".
- For returns: Use "Performance" and historical data.
Step 3: Supplement your answer with key quantitative risk metrics and fund metadata:
- **Alpha:** Indicates outperformance vs. benchmark.
  - Positive alpha: Fund beat its benchmark (good).
  - Negative alpha: Fund lagged benchmark (bad).
- **Sharpe Ratio:** Return per unit of total risk.
  - Positive: Fund outperformed the risk-free rate (good, higher is better).
  - Negative: Underperformed risk-free rate; took risk but lost money.
- **Sortino Ratio:** Like Sharpe but penalizes only downside risk.
  - Positive: Good risk-adjusted performance.
  - Negative: High downside risk or negative returns.
- **Treynor Ratio:** Return per unit of market risk (beta).
  - Positive: Compensated for market risk.
  - Negative: Took market risk but underperformed risk-free rate.
- **Standard Deviation:** Measures volatility.
  - Higher = more volatile, riskier.
  - Lower = more stable.
- **Max Drawdown:** Greatest observed loss from a peak to a trough.
- **Expense Ratio, AUM, Inception Date, etc.**
Step 4: If any risk metric is negative, always explain what that means for the user.
  - E.g., "A negative Sharpe ratio means the fund underperformed safe assets and took on unnecessary risk."
Step 5: Structure your answer:
- **Start with the relevant prospectus summary** for official narrative and disclosures.
- **Follow with a table or bullet points** of quantitative data, including risk metrics and fees.
- **Interpret results in plain language, especially if metrics are negative or unusually high/low.**
- Offer practical insights or suitability if possible (e.g., “This fund may not be suitable for conservative investors given its high volatility.”)
Step 6: If any requested data is missing, reply:  
"That information is not available in the provided context."

Context:
{context}

Question:
{question}
""")

# --- Streamlit page configuration
st.set_page_config(page_title="GenAI Fund Advisor", layout="wide")
st.markdown("<h1 style='color:#38bdf8;'>GenAI Fund Advisor</h1>", unsafe_allow_html=True)
st.markdown("Ask questions about mutual funds, definitions, or risk metrics.")

# --- Dark mode CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #151a21;
            color: #e5e7eb;
            font-family: "Segoe UI", sans-serif;
        }
        h1 {
            color: #38bdf8 !important;
            margin-bottom: 0.2em;
        }
        .stTextInput > div > div > input {
            font-weight: 500;
            background-color: #23272f;
            border: 2px solid #334155;
            border-radius: 8px;
            padding: 0.6em;
            color: #e5e7eb;
        }
        .stButton > button {
            background-color: #38bdf8;
            color: #23272f;
            font-weight: bold;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0ea5e9;
        }
        .custom-answer {
            background-color: #23272f;
            color: #e0e7ef;
            padding: 1em;
            border-radius: 10px;
            border-left: 6px solid #38bdf8;
            font-size: 1.07em;
            margin-bottom: 1em;
        }
        .custom-answer strong {
            color: #38bdf8;
        }
        .custom-explanation {
            background-color: #23272f;
            color: #a3a3a3;
            padding: 1em;
            border-radius: 10px;
            border-left: 6px solid #818cf8;
            font-size: 0.97em;
            margin-bottom: 1em;
        }
        .chat-bubble {
            border-radius: 12px;
            padding: 0.75em 1em;
            margin: 0.5em 0;
            max-width: 95%;
        }
        .chat-bubble.user {
            background-color: #22242b;
            border-left: 5px solid #38bdf8;
        }
        .chat-bubble.ai {
            background-color: #23272f;
            border-left: 5px solid #818cf8;
        }
    </style>
""", unsafe_allow_html=True)

# --- Clear conversation button
if st.button("Clear Conversation"):
    for key in ("memory", "qa_chain"):
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Initialize memory and QA chain
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

# --- Display chat history
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

# --- User query input (uses Enter to submit)
query = st.text_input(
    "Your Question",
    placeholder="E.g. What are the risks of AGG? Or Compare VWELX and VFIAX"
)

# --- PDF state management
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = ""

if query:
    result = st.session_state.qa_chain.invoke({"question": query})
    answer = result.get("answer")
    source_docs = result.get("source_documents", [])
    sources = "\n".join(doc.metadata.get("source", "N/A") for doc in source_docs)

    if not answer or "context does not include" in answer.lower():
        st.warning("No relevant data found for that question.")
        st.session_state.pdf_ready = False
        st.session_state.pdf_path = ""
    else:
        st.markdown("### Response:")

        # --- Split LLM output into Answer and Explanation cards
        if "Explanation:" in answer:
            answer_part, explanation_part = answer.split("Explanation:", 1)
            st.markdown(f"<div class='custom-answer'><strong>Answer:</strong> {answer_part.replace('Answer:', '').strip()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='custom-explanation'><strong>Explanation:</strong> {explanation_part.strip()}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='custom-answer'>{answer.strip()}</div>", unsafe_allow_html=True)

        if sources.strip():
            st.markdown(f"<p><b>Sources:</b><br>{sources}</p>", unsafe_allow_html=True)

        # Only offer PDF if user clicks "Export as PDF"
        if st.button("Export as PDF"):
            class CustomPDF(FPDF):
                def header(self):
                    self.set_font("Arial", size=12)
                    self.cell(0, 10, "GenAI Fund Advisor Response", ln=True, align="C")

            pdf = CustomPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            if "Explanation:" in answer:
                answer_part, explanation_part = answer.split("Explanation:", 1)
                pdf.multi_cell(0, 10, f"Answer:\n{answer_part.replace('Answer:', '').strip()}\n\nExplanation:\n{explanation_part.strip()}")
            else:
                pdf.multi_cell(0, 10, answer.strip())

            if sources:
                pdf.ln(5)
                pdf.multi_cell(0, 10, f"Sources:\n{sources}")

            pdf_path = "response.pdf"
            pdf.output(pdf_path)
            st.session_state.pdf_ready = True
            st.session_state.pdf_path = pdf_path

# Show Download button only if PDF was requested and is ready
if st.session_state.pdf_ready and st.session_state.pdf_path:
    with open(st.session_state.pdf_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="response.pdf")
