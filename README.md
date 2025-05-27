# GenAI Fund Recommendation Tool

## Overview

This project is a proof-of-concept (POC) AI-powered tool designed to assist financial advisors in recommending suitable investment funds based on individual client risk profiles. By integrating public financial data with large language models, the system generates clear, human-readable fund summaries and recommendations tailored to varying risk levels.

## Features

- Collects fund data from trusted public sources such as Yahoo Finance and Morningstar
- Cleans and processes key metrics (e.g., Sharpe Ratio, Expense Ratio, Beta, Returns)
- Integrates with OpenAI's GPT to generate personalized fund recommendations
- Provides a user interface for financial advisors to select risk level and fund category
- Displays AI-generated summaries and justifications to aid advisor decision-making

## Tech Stack

- **Frontend:** Streamlit or React (under evaluation)
- **Backend:** Python, Flask or FastAPI
- **Data Handling:** yfinance, BeautifulSoup, pandas, NumPy
- **AI Integration:** OpenAI API, LangChain (for prompt management)
- **Storage:** CSV/JSON for local data; Azure Blob or AWS S3 (cloud-ready)
- **Version Control:** Git + GitHub

## Modules

1. **Data Acquisition**  
   Collects fund performance metrics, fact sheets, and financial ratios from public sources.

2. **Data Preprocessing**  
   Cleans, filters, and formats data for AI consumption using pandas and NumPy.

3. **GenAI Integration**  
   Sends cleaned fund data to OpenAI’s GPT model and returns personalized recommendations with reasoning.

4. **User Interface**  
   Allows advisors to input client risk level and view fund summaries and recommendations in an interactive dashboard.

## Collaboration with Business Team

- Business team defines client risk profiles and mapping logic
- Provides criteria for fund evaluation and use case scenarios
- Reviews and validates AI-generated summaries and fund suggestions
- Regular syncs are held to align on logic, UX feedback, and prompt refinement

## Getting Started

> This section will be updated when development begins in Week 3. It will include instructions for setting up the environment, installing dependencies, and running the local version of the tool.

## License

This project is currently under internal development for educational and research purposes.
