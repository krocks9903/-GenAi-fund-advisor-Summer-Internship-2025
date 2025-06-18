from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json

# 
target_metrics = {
    "Mean Annual Return",
    "Sharpe Ratio",
    "Treynor Ratio",
    "Standard Deviation",
    "Expense Ratio"
}

def scrape_selected_risk_metrics(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/risk"
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)

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
                    "3y": cols[1].text.strip(),
                    "5y": cols[3].text.strip(),
                    "10y": cols[5].text.strip()
                }

    return results

def print_selected_risk_metrics(ticker, data):
    print(f"\n📊 Selected Risk Metrics for {ticker}\n")
    header = f"{'Metric':<20} {'3Y':<10} {'5Y':<10} {'10Y':<10}"
    print(header)
    print("-" * len(header))
    for metric, values in data.items():
        print(f"{metric:<20} {values['3y']:<10} {values['5y']:<10} {values['10y']:<10}")

if __name__ == "__main__":
    tickers = [
        # Conservative
        "VWELX", "VBTLX", "AGG",
        # Moderate
        "VFIAX", "VTMFX", "PRBLX",
        # Aggressive
        "FSPTX", "VSMAX", "VTIAX", "FCPGX"
    ]

    all_results = {}

    for ticker in tickers:
        print(f"\n⏳ Scraping {ticker}...")
        data = scrape_selected_risk_metrics(ticker)
        if data:
            print_selected_risk_metrics(ticker, data)
            all_results[ticker] = data
        else:
            print(f"⚠️ No matching metrics found for {ticker}.")
        time.sleep(3)  # polite delay

    # Save results to JSON
    with open("fund_risk_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ All fund data saved to 'fund_risk_metrics.json'")
