
---

## Technologies Used

### Backend

- Python 3.11+
- Flask or FastAPI
- BeautifulSoup, Selenium (for scraping)
- pandas, NumPy (data cleaning and transformation)

### AI & Retrieval

- LangChain
- OpenAI GPT-4 API
- FAISS or Azure Cognitive Search
- PromptLayer (optional for managing prompt logs)

### Frontend

- Streamlit (MVP UI)
- Optionally React (under evaluation for future scalability)

### Data Sources

- [Yahoo Finance](https://finance.yahoo.com/)
- [Morningstar](https://www.morningstar.com/)
- [Fidelity](https://www.fidelity.com/)

---

## Modules

### 1. Data Acquisition

Scrapes financial metrics and fund profiles for mutual funds. Focuses on metrics such as:

- Sharpe Ratio
- Sortino Ratio
- Mean Annual Return
- Alpha / Beta
- Expense Ratio
- Max Drawdown
- Treynor Ratio
- Standard Deviation
- # of Years Up/Down (when available)

### 2. Data Preprocessing

Cleans and transforms scraped data into a structured format (JSON or CSV) for RAG indexing and LLM input.

### 3. Vector Store & Retrieval

Fund documents or summaries are chunked, embedded (e.g., with `text-embedding-ada-002`), and stored in a vector database (FAISS or Azure Vector DB). Used by LangChain retrievers.

### 4. RAG + GPT Integration

Queries are matched with stored vector chunks and passed to OpenAI's GPT-4 for generating responses. Prompt templates are used to enforce format, style, and explainability.

### 5. Recommendation Interface

The front-end dashboard accepts user-selected risk levels (e.g., Low / Medium / High) and renders fund recommendations ranked by metrics relevant to that profile.

---

## Collaboration with Business Team

- Business team defines client risk profiles and ranking strategies
- Technical team implements metric extraction and scoring engine
- Joint evaluation of model outputs to improve prompt design and data sourcing
- Weekly syncs held for use-case alignment, bug tracking, and UI/UX feedback

---

## Setup Instructions (Coming Soon)

This section will include:

- Environment setup
- API key integration (OpenAI, vector DB)
- Running the Streamlit app locally
- How to test and validate recommendations

---

## License

This project is under active research and development. It is currently closed-source and intended for internal, educational, and non-commercial use.

