from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json
import os

# Only extract these metrics
target_metrics = {
    "Mean Annual Return",
    "Sharpe Ratio",
    "Treynor Ratio",
    "# of Years Up",
    "Standard Deviation",
    "Sortino Ratio",
    "Expense Ratio",
    "Max Drawdown"
}

def scrape_selected_risk_metrics(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/risk"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(4)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    rows = soup.find_all("tr", class_="yf-1m2lefb useStripingColumn")
    results = {}

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 7:
            metric_name = cols[0].text.strip()
            if metric_name in target_metrics:
                results[metric_name] = {
                    "3y_fund": cols[1].text.strip(),
                    "3y_avg": cols[2].text.strip(),
                    "5y_fund": cols[3].text.strip(),
                    "5y_avg": cols[4].text.strip(),
                    "10y_fund": cols[5].text.strip(),
                    "10y_avg": cols[6].text.strip()
                }

    return results

def print_and_save_metrics(ticker, data):
    print(f"\nSelected Risk Metrics for {ticker}\n")
    header = f"{'Metric':<20} {'3Y Fund':<10} {'3Y Avg':<10} {'5Y Fund':<10} {'5Y Avg':<10} {'10Y Fund':<10} {'10Y Avg':<10}"
    print(header)
    print("-" * len(header))
    for metric, values in data.items():
        print(f"{metric:<20} {values['3y_fund']:<10} {values['3y_avg']:<10} {values['5y_fund']:<10} {values['5y_avg']:<10} {values['10y_fund']:<10} {values['10y_avg']:<10}")

    # Save to JSON
    if not os.path.exists("output"):
        os.mkdir("output")

    filepath = f"output/{ticker}.json"
    with open(filepath, "w") as f:
        json.dump({ticker: data}, f, indent=2)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    while True:
        ticker = input("Enter mutual fund ticker (or 'exit' to quit): ").strip().upper()
        if ticker == 'EXIT':
            break
        data = scrape_selected_risk_metrics(ticker)
        if data:
            print_and_save_metrics(ticker, data)
        else:
            print(f"No valid metrics found for {ticker}")
