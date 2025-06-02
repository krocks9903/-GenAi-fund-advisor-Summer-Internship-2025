# Mutual Fund Risk Metrics Scraper

## Overview

This script scrapes publicly available risk and performance statistics for mutual funds from Yahoo Finance. The goal is to extract key metrics over 3-year, 5-year, and 10-year periods for use in downstream analysis, such as LLM-based recommendation systems or financial dashboards.

The output is saved as structured JSON, mapping each fund's ticker symbol to its relevant performance metrics and benchmark comparisons.

---

## Sample Output Format

Each fund is stored as a JSON object with time-based values for both the fund and its category average. For example:

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
