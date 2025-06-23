import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
from fpdf import FPDF
import time

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
st.set_page_config(
    page_title="WealthAI", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💰"
)

# Custom CSS for modern design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header-section {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .header-subtitle {
            font-size: 1.2rem;
            color: #6b7280;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        .chat-container {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
        }
        
        .chat-message {
            margin: 1rem 0;
            padding: 1.2rem;
            border-radius: 12px;
            animation: slideIn 0.3s ease-out;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: 2rem;
            border-bottom-right-radius: 4px;
        }
        
        .ai-message {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            color: #1f2937;
            margin-right: 2rem;
            border-bottom-left-radius: 4px;
            border-left: 4px solid #667eea;
        }
        
        .input-container {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
        }
        
        .stTextArea > div > div > textarea {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            background: #f9fafb;
            color: #000000 !important;
            transition: all 0.3s ease;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            background: #ffffff;
        }
        
        .stTextArea > div > div > textarea::placeholder {
            color: #9ca3af !important;
            opacity: 1;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .response-card {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
        }
        
        .sources-section {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid #e2e8f0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #f59e0b;
            box-shadow: 0 2px 10px rgba(245, 158, 11, 0.1);
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .feature-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .feature-card:hover::before {
            left: 100%;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h3 {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .feature-card p {
            color: #764ba2;
            font-weight: 400;
            line-height: 1.5;
        }
        
        .clear-button {
            background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        }
        
        .clear-button:hover {
            background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-container">
    <div class="header-section">
        <h1 class="header-title">💰 GenAI Fund Advisor</h1>
        <p class="header-subtitle">Your intelligent companion for mutual fund analysis and investment decisions</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards with click functionality
st.markdown("""
<div class="main-container">
    <div class="feature-grid">
        <div class="feature-card" onclick="setQuery('What is Sharpe ratio and how is it calculated?')">
            <div class="feature-icon">📊</div>
            <h3>Risk Analysis</h3>
            <p>Comprehensive evaluation of Sharpe, Sortino, and Treynor ratios</p>
        </div>
        <div class="feature-card" onclick="setQuery('Compare VWELX vs VFIAX performance metrics')">
            <div class="feature-icon">🔍</div>
            <h3>Fund Comparison</h3>
            <p>Side-by-side analysis of mutual fund performance metrics</p>
        </div>
        <div class="feature-card" onclick="setQuery('Recommend a low-risk long-term investment fund')">
            <div class="feature-icon">💡</div>
            <h3>Smart Recommendations</h3>
            <p>Personalized investment suggestions based on your risk profile</p>
        </div>
    </div>
</div>

<script>
function setQuery(query) {
    // Find the textarea element and set its value
    const textArea = parent.document.querySelector('textarea[aria-label=""]');
    if (textArea) {
        textArea.value = query;
        textArea.dispatchEvent(new Event('input', { bubbles: true }));
        textArea.focus();
        
        // Scroll to the input area
        textArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}
</script>
""", unsafe_allow_html=True)

# Clear conversation button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🗑️ Clear Conversation", key="clear_btn"):
        for key in ("memory", "qa_chain"):
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Custom prompt (unchanged)
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
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
        if msg.type == "human":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑‍💼 You:</strong><br>{msg.content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>🤖 WealthAI:</strong><br>{msg.content}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# User input section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="input-container">', unsafe_allow_html=True)

st.markdown("### 💬 Ask Your Question")
query = st.text_area(
    "",
    placeholder="💡 Try asking: 'What is the Sortino Ratio?' or 'Compare PRBLX and FSPTX performance'",
    height=120,
    key="user_query"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    ask_button = st.button("🚀 Get Analysis", key="ask_btn", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process query
if ask_button and query:
    with st.spinner("🔍 Analyzing your request..."):
        result = st.session_state.qa_chain.invoke({"question": query})
        answer = result.get("answer")
        source_docs = result.get("source_documents", [])
        sources = "\n".join(doc.metadata.get("source", "N/A") for doc in source_docs)

        st.markdown(f"""
        <div class="main-container">
            <div class="response-card">
                <h3>📈 Analysis Results</h3>
                <div style="margin-top: 1rem; line-height: 1.6;">{answer}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if sources.strip():
            st.markdown(f"""
            <div class="main-container">
                <div class="sources-section">
                    <h4>📚 Sources</h4>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #6b7280;">{sources}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Export as PDF", key="export_btn"):
                class CustomPDF(FPDF):
                    def header(self):
                        self.set_font("Arial", "B", 16)
                        self.cell(0, 10, "GenAI Fund Advisor - Analysis Report", ln=True, align="C")
                        self.ln(10)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font("Arial", "I", 8)
                        self.cell(0, 10, f"Page {self.page_no()}", align="C")

                pdf = CustomPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Question:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, query)
                pdf.ln(5)

                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Analysis:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, answer)

                if sources:
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, "Sources:", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, sources)

                pdf_path = "fund_analysis_report.pdf"
                pdf.output(pdf_path)

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download Report", 
                        f, 
                        file_name="fund_analysis_report.pdf",
                        mime="application/pdf"
                    )
# Footer
st.markdown("""
<div class="main-container" style="text-align: center; margin-top: 3rem;">
    <p style="color: #6b7280; font-size: 0.9rem;">
        🔒 Powered by Azure OpenAI • Built with Streamlit • Your financial data is secure
    </p>
</div>
""", unsafe_allow_html=True)
