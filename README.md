# GenAI Fund Recommendation Platform

## Project Description

The GenAI Fund Recommendation Platform is a proof-of-concept tool built to help financial advisors generate **personalized mutual fund recommendations** based on a client's risk profile. Using public financial data and combining it with Large Language Models through a RAG (Retrieval-Augmented Generation) approach, the platform delivers explainable and data-backed fund insights.

The project blends **traditional financial metrics** with **modern generative AI pipelines**, letting users explore fund suggestions aligned to different risk levels: Conservative, Moderate, and Aggressive.

The goal is to **enhance and automate** the advisor workflow by turning raw financial data into personalized, interpretable, and traceable recommendations.

---

## Technologies Used

### Backend

* Python 3.11+
* Flask or FastAPI
* BeautifulSoup (for web scraping)
* pandas, NumPy (for data processing)

### AI & Retrieval

* LangChain (RAG orchestration)
* OpenAI GPT-4 API
* FAISS or Azure Cognitive Search (vector retrieval)
* PromptLayer (optional prompt tracking)

### Frontend

* Streamlit (interactive MVP dashboard)
* React (planned for future scalable frontend)

### Data Sources

* [Yahoo Finance](https://finance.yahoo.com/)
* [Morningstar](https://www.morningstar.com/)
* [Fidelity](https://www.fidelity.com/)

These sources provide access to core metrics like Sharpe Ratio, Sortino Ratio, and Max Drawdown.

---

## Modules

### 1. Data Acquisition

Scrapes mutual fund data from public sources and collects key financial risk/return metrics:

* Mean Annual Return
* Sharpe Ratio
* Sortino Ratio
* Alpha & Beta
* Expense Ratio
* Max Drawdown
* Treynor Ratio
* Standard Deviation

### 2. Data Preprocessing

Scraped HTML content is cleaned and parsed into a **normalized JSON format**, with consistent field names and types. This structured format feeds the vector store and LLM pipeline.

### 3. Vector Store & Retrieval

The cleaned JSON data is embedded using OpenAI's `text-embedding-ada-002`, and indexed using **FAISS** or **Azure Cognitive Search**. LangChain retrieves contextually relevant snippets based on user queries.

### 4. RAG + GPT Integration

LangChain orchestrates the full pipeline:

* User query (e.g., "Top conservative funds") is matched to relevant data
* Retrieved chunks are passed into a prompt
* GPT-4 generates a **personalized and explainable** response using both structured data and natural language reasoning

### 5. Recommendation Interface

The Streamlit interface allows users (e.g., advisors or interns) to:

* Input a risk level (Conservative / Moderate / Aggressive)
* View fund metrics and AI-generated recommendations
* Export outputs (PDF coming soon)

---

## Collaboration with Business Team

* The business team defines the logic for risk categories
* They review and validate scraped data and metrics
* They provide real-world use cases and ensure outputs are usable in advisor workflows
* Weekly syncs help iterate quickly and refine priorities

---

## Setup Instructions (Coming Soon)

* Clone the repo and set up virtual environment
* Install dependencies
* Configure API keys (.env file for OpenAI, Azure, etc.)
* Run the scraper and generate JSON
* Start Streamlit app to test recommendations

---

## Sample Prompts

These are example queries that financial advisors or end users might enter into the interface:

* "Show the top 3 conservative mutual funds for a retired investor"
* "Compare VFIAX and VWELX on risk-adjusted return over 10 years"
* "Explain why VSMAX is considered an aggressive option"
* "Which fund had the lowest max drawdown among moderate-risk funds?"
* "Summarize performance of FCPGX including Sharpe and Treynor ratios"

These prompts are processed by the LLM through LangChain's RAG flow, where it retrieves relevant fund metrics and generates a tailored response.

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
```


