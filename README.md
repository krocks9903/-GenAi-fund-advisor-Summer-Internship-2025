# GenAI Fund Recommendation Platform

## Project Description

The GenAI Fund Recommendation Platform is a proof-of-concept application that aims to assist financial advisors in generating personalized mutual fund recommendations based on a client's risk profile. By leveraging public financial data sources and combining them with large language models via a RAG (Retrieval-Augmented Generation) architecture, this tool delivers explainable, data-backed investment insights.

This project merges traditional financial metric analysis with modern generative AI and retrieval pipelines, allowing users to surface relevant, context-aware fund summaries across various risk tolerance levels (Low, Medium, High).

The core goal is to automate and enhance the advisor's workflow by converting raw financial data into personalized, interpretable, and auditable recommendations.

---

## Technologies Used

### Backend

- Python 3.11+
- Flask or FastAPI
- BeautifulSoup, Selenium (for web scraping)
- pandas, NumPy (for data cleaning and metric processing)

### AI & Retrieval

- LangChain (RAG pipeline and orchestration)
- OpenAI GPT-4 API
- FAISS or Azure Cognitive Search (vector store for retrieval)
- PromptLayer (optional logging of prompts and completions)

### Frontend

- Streamlit (MVP interactive dashboard)
- React (future-ready scalable interface, under evaluation)

### Data Sources

- [Yahoo Finance](https://finance.yahoo.com/)
- [Morningstar](https://www.morningstar.com/)
- [Fidelity](https://www.fidelity.com/)

These sources provide access to key fund performance and risk metrics such as Sharpe Ratio, Standard Deviation, and Treynor Ratio.

---

## Modules

### 1. Data Acquisition

Scrapes financial and risk metrics for mutual funds from public websites. Currently focused on:

- Sharpe Ratio  
- Sortino Ratio  
- Mean Annual Return  
- Alpha / Beta  
- Expense Ratio  
- Max Drawdown  
- Treynor Ratio  
- Standard Deviation  
- Number of Years Up/Down (if available)

### 2. Data Preprocessing

Raw HTML content is parsed and cleaned using `BeautifulSoup`. Extracted data is normalized into structured JSON format with consistent naming and typing. This makes the data usable for vector indexing and LLM input.

### 3. Vector Store & Retrieval

Processed fund data (including summaries, metric explanations, etc.) is embedded using OpenAI's `text-embedding-ada-002` and indexed using FAISS or Azure's Cognitive Search. Retrieval is optimized for similarity-based matching at query time.

### 4. RAG + GPT Integration

LangChain handles the prompt pipeline:
- The user query (e.g., "Show best low-risk funds") is matched to relevant vector chunks
- Retrieved documents are injected into prompt templates
- GPT-4 generates a personalized, explainable recommendation based on both raw data and learned patterns

### 5. Recommendation Interface

A simple Streamlit UI allows users (e.g., financial advisors) to:
- Input client risk level (Low / Medium / High)
- Trigger fund comparison
- View structured fund data and LLM-generated recommendations

---

## Collaboration with Business Team

- Business stakeholders define and refine the client risk segmentation logic
- They provide the ranking criteria and validate scraped metrics
- Collaboration ensures that the generated recommendations align with practical expectations
- Weekly working sessions are used for feedback, iteration, and logic validation

---

## Setup Instructions (Coming Soon)

This section will include:

- How to clone and set up the environment
- How to install dependencies
- Setting up API keys (e.g., OpenAI, vector DB)
- Running the scraper and data preprocessor
- Starting the Streamlit app
- Using test prompts and validating RAG outputs

---

## Example Output (JSON)

Sample output for a fund (e.g., `VGTSX`) from the scraper:

```json
{
  "VGTSX": {
    "Mean Annual Return": {
      "3y_fund": "0.75",
      "3y_avg": "0.01",
      "5y_fund": "0.92",
      "5y_avg": "0.01",
      "10y_fund": "0.5",
      "10y_avg": "0.01"
    },
    "Standard Deviation": {
      "3y_fund": "16.33",
      "3y_avg": "0.18",
      "5y_fund": "15.46",
      "5y_avg": "0.15",
      "10y_fund": "15.09",
      "10y_avg": "0.15"
    },
    "Sharpe Ratio": {
      "3y_fund": "0.27",
      "3y_avg": "0",
      "5y_fund": "0.53",
      "5y_avg": "0.01",
      "10y_fund": "0.26",
      "10y_avg": "0"
    },
    "Treynor Ratio": {
      "3y_fund": "3.16",
      "3y_avg": "0.07",
      "5y_fund": "7.38",
      "5y_avg": "0.09",
      "10y_fund": "2.93",
      "10y_avg": "0.05"
    }
  }
}
