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
st.set_page_config(
    page_title="ORION: Optimal Risk-Investment Outreach Navigator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
.stApp { background: #eaf4fb; color: #1a2a3a; font-family: 'Segoe UI', 'Arial', sans-serif;}
.stSidebar {background: #1c3158 !important;}
.logo-row {display: flex; align-items: center; margin-bottom: 0.6em;}
.logo-row img {height: 64px; margin-right: 0.8em;}
h1 {font-size: 2.3em !important; font-weight: bold !important; color: #1c3158 !important; margin-bottom: 0.09em;}
.stTextInput > div > div > input {
    font-weight: 500; background-color: #f6fafd; border: 2px solid #225da3;
    border-radius: 10px; padding: 0.7em; color: #222;
}
.stButton > button {background-color: #225da3; color: #fff; font-weight: 600; border: none; border-radius: 10px;}
.stButton > button:hover {background-color: #497cf7;}
.custom-answer {background: #f2f8fd; color: #15203b; padding: 1.18em 1.22em; border-radius: 11px; border-left: 5px solid #225da3;}
.custom-answer strong {color: #225da3;}
.custom-explanation {background: #f2f6fb; color: #1c3158; padding: 1.1em 1.22em; border-radius: 11px; border-left: 5px solid #497cf7;}
.footer {color: #6c7a93; text-align: center; font-size: 0.99em; padding: 2em 0 1em 0; margin-top: 3.5em;}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar with logo and navigation ----
logo_path = "Scripts/infosys_logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=160)
st.sidebar.markdown(
    "<span style='font-size:1.15em; font-weight:600; color:#1c3158;'>ORION: Optimal Risk-Investment Outreach Navigator</span>",
    unsafe_allow_html=True,
)

sidebar_options = [
    "👤 Client Names",
    "💬 GenAI Fund Page",
    "📄 Documents",
    "⚙️ Settings"
]
selected = st.sidebar.radio("Menu", sidebar_options, index=1, label_visibility="collapsed")

# --- Track documents (sources) in session ---
if "documents" not in st.session_state:
    st.session_state["documents"] = set()

# ---- Routing ----

if selected == "👤 Client Names":
    st.markdown("<h1>Client Names</h1>", unsafe_allow_html=True)
    st.markdown("<p>List your clients here. You can make this section interactive (search, click to expand info, etc.).</p>", unsafe_allow_html=True)

elif selected == "📄 Documents":
    st.markdown("<h1>Documents</h1>", unsafe_allow_html=True)
    docs = st.session_state.get("documents", set())
    if not docs:
        st.info("No sources have been referenced yet in this session.")
    else:
        for idx, src in enumerate(sorted(docs), 1):
            st.markdown(f"{idx}. {src}")

elif selected == "⚙️ Settings":
    st.markdown("<h1>Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p>User preferences, login management, appearance settings, etc.</p>", unsafe_allow_html=True)

elif selected == "💬 GenAI Fund Page":
    st.markdown("<h1>ORION: Optimal Risk-Investment Outreach Navigator</h1>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.09em;color:#1c3158;margin-bottom:1.7em;'>Ask questions about mutual funds, definitions, or risk metrics.</div>", unsafe_allow_html=True)

    # ---- Clear Conversation button
    col1, col2 = st.columns([2, 6])
    with col1:
        if st.button("Clear Conversation"):
            for key in ("memory", "qa_chain"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ---- Initialize memory and QA chain ----
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

    # ---- Example prompt panel if no chat yet ----
    if not st.session_state.memory.chat_memory.messages:
        st.markdown(
            """
            <div style="background: #f4f9fd; border-left: 5px solid #225da3; border-radius: 9px; padding: 1.1em 1.2em 1.1em 1.35em; margin-bottom:1.5em; font-size:1.06em;">
            <b>Example questions:</b>
            <ul class='example-list' style="margin-top:0.3em;">
                <li>What are the risks of AGG?</li>
                <li>Compare VWELX and VFIAX on performance</li>
                <li>What does a negative Sharpe Ratio mean?</li>
                <li>Which fund has the highest alpha over 5 years?</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---- Display chat history ----
    if st.session_state.memory.chat_memory.messages:
        st.markdown("<h3 style='margin-top:2em;margin-bottom:0.7em;'>Chat History</h3>", unsafe_allow_html=True)
        for msg in st.session_state.memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "GenAI"
            bubble_class = "user" if msg.type == "human" else "ai"
            st.markdown(f"""
            <div class="chat-bubble {bubble_class}">
                <strong>{role}:</strong><br>{msg.content}
            </div>
            """, unsafe_allow_html=True)

    # ---- User query input (uses Enter to submit)
    query = st.text_input(
        "Your Question",
        placeholder="E.g. What are the risks of AGG? Or Compare VWELX and VFIAX"
    )

    # ---- PDF state management
    if "pdf_ready" not in st.session_state:
        st.session_state.pdf_ready = False
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = ""

    if query:
        result = st.session_state.qa_chain.invoke({"question": query})
        answer = result.get("answer")
        source_docs = result.get("source_documents", [])
        # Collect all sources into session documents
        for doc in source_docs:
            source = doc.metadata.get("source", "N/A")
            if source and source != "N/A":
                st.session_state["documents"].add(source)
        sources = "\n".join(doc.metadata.get("source", "N/A") for doc in source_docs)

        if not answer or "context does not include" in answer.lower():
            st.warning("No relevant data found for that question.")
            st.session_state.pdf_ready = False
            st.session_state.pdf_path = ""
        else:
            st.markdown("<h3 style='margin-top:2.5em;'>Response</h3>", unsafe_allow_html=True)
            # --- Split LLM output into Answer and Explanation cards
            if "Explanation:" in answer:
                answer_part, explanation_part = answer.split("Explanation:", 1)
                st.markdown(f"<div class='custom-answer'><strong>Answer:</strong> {answer_part.replace('Answer:', '').strip()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='custom-explanation'><strong>Explanation:</strong> {explanation_part.strip()}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='custom-answer'>{answer.strip()}</div>", unsafe_allow_html=True)

            # --- PDF export and sources
            colpdf, colsources = st.columns([2, 3])
            with colpdf:
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
            with colsources:
                if sources.strip():
                    st.markdown(f"<div style='font-size:1.01em;color:#1c3158;'><b>Sources:</b><br>{sources}</div>", unsafe_allow_html=True)

            if st.session_state.pdf_ready and st.session_state.pdf_path:
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="response.pdf")

# ---- Footer ----
st.markdown(
    "<div class='footer'>© 2025 Infosys GenAI Fund Advisor &nbsp;|&nbsp; Powered by OpenAI and LangChain</div>",
    unsafe_allow_html=True
)
