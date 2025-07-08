import os
import sys
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Third-party imports with error handling
try:
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    import streamlit as st
    from fpdf import FPDF
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with: pip install -r requirements.txt")
    sys.exit(1)

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orion_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Failed to load environment variables: {e}")
    st.error("Failed to load environment configuration. Please check your .env file.")

# --- Streamlit page config ---
st.set_page_config(
    page_title="ORION: Optimal Risk-Investment Outreach Navigator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💼"
)

# --- Session state initialization with error handling ---
def initialize_session_state():
    """Initialize session state variables with proper defaults"""
    defaults = {
        "documents": set(),
        "active_page": "💬 GenAI Fund Page",
        "feedback_log": [],
        "conversation_history": [],
        "last_query": "",
        "error_log": [],
        "initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Environment validation ---
def validate_environment() -> bool:
    """Validate that all required environment variables are present"""
    required_vars = [
        "AZURE_API_BASE",
        "AZURE_API_KEY", 
        "AZURE_EMBED_DEPLOYMENT",
        "AZURE_EMBED_VERSION",
        "AZURE_CHAT_DEPLOYMENT",
        "AZURE_CHAT_VERSION"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error(f"Missing environment variables: {missing_vars}")
        return False
    
    return True

# --- Initialize components with error handling ---
@st.cache_resource
def initialize_embeddings():
    """Initialize Azure OpenAI embeddings with caching and error handling"""
    try:
        embedding = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
            api_version=os.getenv("AZURE_EMBED_VERSION"),
            chunk_size=1000,
        )
        logger.info("Embeddings initialized successfully")
        return embedding
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None

@st.cache_resource
def initialize_vectorstore():
    """Initialize FAISS vectorstore with error handling"""
    try:
        embedding = initialize_embeddings()
        if embedding is None:
            return None
            
        if not os.path.exists("faiss_index_fund_data"):
            st.error("FAISS index directory not found. Please ensure the index is properly created.")
            logger.error("FAISS index directory not found")
            return None
            
        vectorstore = FAISS.load_local(
            "faiss_index_fund_data",
            embedding,
            allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}")
        st.error(f"Failed to load vectorstore: {str(e)}")
        return None

@st.cache_resource
def initialize_llm():
    """Initialize Azure OpenAI chat model with error handling"""
    try:
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            model="gpt-4",  # Fixed model name
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_CHAT_VERSION"),
            temperature=0,
            max_tokens=1000,  # Added token limit
            timeout=30,  # Added timeout
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        st.error(f"Failed to initialize language model: {str(e)}")
        return None

# --- Enhanced prompt template ---
def get_custom_prompt():
    """Get the custom prompt template"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""INSTRUCTION: First, provide a direct and concise answer to the user's question in 2-3 sentences and label it as 'Answer:'. After that, clearly label and provide your full explanation or reasoning as 'Explanation:' based on the chain-of-thought steps.

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
- **Sharpe Ratio:** Return per unit of total risk.
- **Sortino Ratio:** Penalizes only downside risk.
- **Treynor Ratio:** Return per unit of market risk (beta).
- **Standard Deviation:** Measures volatility.
- **Max Drawdown:** Greatest observed loss.
- **Expense Ratio, AUM, Inception Date, etc.**
Step 4: If any risk metric is negative, explain its meaning.
Step 5: Structure your answer:
- Start with prospectus summary
- Follow with key metrics
- Interpret results clearly
Step 6: If any requested data is missing, reply:
"That information is not available in the provided context."

Context:
{context}

Question:
{question}

Answer:"""
    )

# --- Enhanced CSS ---
def load_custom_css():
    """Load custom CSS with improved styling"""
    st.markdown("""
    <style>
    body, .stApp { 
        background: #eaf4fb; 
        color: #1c3158; 
        font-family: 'Segoe UI', sans-serif; 
    }
    .stSidebar { 
        background-color: #1c3158 !important; 
    }
    .main-header {
        color: #1c3158;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(135deg, #1c3158, #2d5aa0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .search-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .response-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .custom-answer {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1c3158;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .custom-explanation {
        background: #f0f8ff;
        padding: 1rem;
        border-left: 4px solid #2d5aa0;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .example-hint {
        font-size: 0.96rem;
        color: #1c3158;
        margin-bottom: 1rem;
        font-style: italic;
        text-align: center;
    }
    .error-message {
        background: #ffe6e6;
        color: #d32f2f;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #d32f2f;
        margin: 1rem 0;
    }
    .success-message {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
    .feedback-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        justify-content: center;
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def process_query(query: str, qa_chain) -> Dict[str, Any]:
    """Process user query with fallback to LLM if vector context fails"""
    try:
        if not query or not query.strip():
            return {"error": "Please enter a valid query"}
        
        # Add query to history
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "type": "user"
        })

        with st.spinner("Analyzing your query..."):
            result = qa_chain.invoke({"question": query})
        
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        # If no documents were used, fallback to LLM only
        if not source_docs:
            fallback_prompt = f"""You are a mutual fund expert assistant. The user asked:

{query}

Provide a helpful and concise answer based on your general knowledge of mutual funds. If you're unsure, say so."""
            llm = st.session_state.qa_chain.llm
            fallback_response = llm.invoke(fallback_prompt)
            answer = f"**Note:** This answer is based on general knowledge, as no supporting documents were found.**\n\n{fallback_response}"
        
        # Save answer to history
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": answer,
            "type": "assistant",
            "source_count": len(source_docs)
        })

        return {
            "answer": answer,
            "source_docs": source_docs,
            "success": True
        }

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"Query processing error: {e}")
        st.session_state.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "query": query
        })
        return {"error": error_msg}

# --- Feedback handling ---
def handle_feedback(query: str, feedback_type: str):
    """Handle user feedback with logging"""
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "feedback": feedback_type,
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    st.session_state.feedback_log.append(feedback_entry)
    logger.info(f"Feedback received: {feedback_type} for query: {query[:50]}...")
    
    # Show appropriate message
    if feedback_type == "positive":
        st.success("✅ Thank you for your positive feedback!")
    else:
        st.warning("⚠️ Thank you for your feedback. We'll work on improving this.")

# --- Initialize everything ---
def initialize_app():
    """Initialize the application with comprehensive error handling"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Validate environment
        if not validate_environment():
            st.stop()
        
        # Initialize components
        vectorstore = initialize_vectorstore()
        llm = initialize_llm()
        
        if vectorstore is None or llm is None:
            st.error("Failed to initialize core components. Please check your configuration.")
            st.stop()
        
        # Initialize memory and chain
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        if "qa_chain" not in st.session_state:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 relevant documents
            )
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": get_custom_prompt()},
                output_key="answer"
            )
        
        st.session_state.initialized = True
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        st.error(f"Application initialization failed: {str(e)}")
        st.stop()

# --- Main application ---
def main():
    """Main application function"""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Initialize the app
        initialize_app()
        
        # Sidebar
        logo_path = "Scripts/infosys_logo.png"
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_container_width=True)
        
        st.sidebar.markdown(
            "<span style='font-size:1.15em; font-weight:600; color:#ffffff;'>ORION: Optimal Risk-Investment Outreach Navigator</span>",
            unsafe_allow_html=True,
        )
        
        # Sidebar navigation
        sidebar_options = [
            "👤 Client Names",
            "💬 GenAI Fund Page", 
            "📄 Documents",
            "📊 Analytics",
            "⚙️ Settings"
        ]
        
        selected = st.sidebar.radio(
            "Menu",
            sidebar_options,
            index=sidebar_options.index(st.session_state.get("active_page", "💬 GenAI Fund Page")),
            label_visibility="collapsed"
        )
        st.session_state["active_page"] = selected
        
        # Main content area
        if selected == "💬 GenAI Fund Page":
            render_chat_page()
        elif selected == "📄 Documents":
            render_documents_page()
        elif selected == "📊 Analytics":
            render_analytics_page()
        elif selected == "⚙️ Settings":
            render_settings_page()
        else:
            st.info("This page is under development.")
        
        # Footer
        st.markdown("""
        <div class='footer'>
            © 2025 Infosys GenAI Fund Advisor | Powered by Azure OpenAI and LangChain
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Main application error: {e}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")

def render_chat_page():
    """Render the main chat interface"""
    st.markdown("<h1 class='main-header'>ORION: Optimal Risk-Investment Outreach Navigator</h1>", 
                unsafe_allow_html=True)
    
    # Search container
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        st.markdown("<div class='example-hint'>💡 Try: \"Compare VFIAX and VWELX\", \"What is max drawdown?\", \"Top 3 low-risk funds\"</div>", 
                   unsafe_allow_html=True)
        
        # Input and clear button
        col_input, col_clear = st.columns([8, 1])
        
        with col_input:
            query = st.text_input(
                "Search", 
                placeholder="🔍 Ask your fund question…", 
                key="chat_input", 
                label_visibility="collapsed"
            )
        
        with col_clear:
            if st.button("🗑️", help="Clear Conversation", use_container_width=True):
                # Clear conversation state
                for key in ("memory", "qa_chain", "conversation_history"):
                    if key in st.session_state:
                        if key == "conversation_history":
                            st.session_state[key] = []
                        else:
                            del st.session_state[key]
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if query and query.strip():
        if not st.session_state.get("initialized", False):
            st.error("Application not properly initialized. Please refresh the page.")
            return
        
        result = process_query(query, st.session_state.qa_chain)
        
        if "error" in result:
            st.markdown(f'<div class="error-message">❌ {result["error"]}</div>', 
                       unsafe_allow_html=True)
        else:
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])
            
            # Add source documents to session state
            for doc in source_docs:
                source = doc.metadata.get("source", "N/A")
                if source and source != "N/A":
                    st.session_state["documents"].add(source)
            
            # Display response
            if not answer or "context does not include" in answer.lower():
                st.warning("⚠️ No relevant data found for that question. Please try rephrasing your query.")
            else:
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("<h3 style='color:#1c3158; margin-bottom:1rem;'>📋 Response</h3>", 
                           unsafe_allow_html=True)
                
                # Parse and display answer
                if "Explanation:" in answer:
                    parts = answer.split("Explanation:", 1)
                    answer_part = parts[0].replace("Answer:", "").strip()
                    explanation_part = parts[1].strip()
                    
                    st.markdown(f'<div class="custom-answer"><strong>Answer:</strong> {answer_part}</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="custom-explanation"><strong>Explanation:</strong> {explanation_part}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="custom-answer">{answer}</div>', 
                               unsafe_allow_html=True)
                
                # Feedback section
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    if st.button("👍 Helpful", key="positive_feedback", use_container_width=True):
                        handle_feedback(query, "positive")
                
                with feedback_col2:
                    if st.button("👎 Not Helpful", key="negative_feedback", use_container_width=True):
                        handle_feedback(query, "negative")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Update last query
                st.session_state["last_query"] = query

def render_documents_page():
    """Render documents page"""
    st.markdown("<h1 class='main-header'>📄 Documents</h1>", unsafe_allow_html=True)
    
    if st.session_state.get("documents"):
        st.markdown("### Referenced Documents")
        for doc in st.session_state["documents"]:
            st.markdown(f"- {doc}")
    else:
        st.info("No documents have been referenced yet. Start a conversation to see referenced documents.")

def render_analytics_page():
    """Render analytics dashboard"""
    st.markdown("<h1 class='main-header'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Conversation stats
    conversation_count = len(st.session_state.get("conversation_history", []))
    feedback_count = len(st.session_state.get("feedback_log", []))
    error_count = len(st.session_state.get("error_log", []))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metrics-card">
            <h3>💬 Conversations</h3>
            <h2>{conversation_count}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metrics-card">
            <h3>👍 Feedback</h3>
            <h2>{feedback_count}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metrics-card">
            <h3>⚠️ Errors</h3>
            <h2>{error_count}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Recent queries
    if st.session_state.get("conversation_history"):
        st.markdown("### Recent Queries")
        for item in st.session_state.conversation_history[-5:]:
            if item["type"] == "user":
                st.markdown(f"- **{item['timestamp'][:19]}**: {item['query']}")

def render_settings_page():
    """Render settings page"""
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    # System status
    st.markdown("### System Status")
    
    status_items = [
        ("Environment Variables", validate_environment()),
        ("Vector Store", st.session_state.get("initialized", False)),
        ("Language Model", st.session_state.get("initialized", False)),
    ]
    
    for item, status in status_items:
        status_icon = "✅" if status else "❌"
        st.markdown(f"{status_icon} {item}")
    
    # Clear data options
    st.markdown("### Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Conversation History", use_container_width=True):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
    
    with col2:
        if st.button("Clear Feedback Log", use_container_width=True):
            st.session_state.feedback_log = []
            st.success("Feedback log cleared!")
    
    # Debug information
    with st.expander("Debug Information"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Error Log:", st.session_state.get("error_log", []))

if __name__ == "__main__":
    main()