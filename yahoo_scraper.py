from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

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

def fetch_fund_risk_statistics(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/risk"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument("user-agent=Mozilla/5.0")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        print(f"Loading {url} ...")
        driver.get(url)
        time.sleep(3)  # Short fixed delay (faster than WebDriverWait here)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.find_all("tr", class_="yf-1m2lefb useStripingColumn")

        metrics = {}
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 7:
                metric_name = cols[0].text.strip()
                if metric_name in target_metrics:
                    metrics[metric_name] = {
                        "3y_fund": cols[1].text.strip(),
                        "3y_avg": cols[2].text.strip(),
                        "5y_fund": cols[3].text.strip(),
                        "5y_avg": cols[4].text.strip(),
                        "10y_fund": cols[5].text.strip(),
                        "10y_avg": cols[6].text.strip()
                    }

        return metrics

    finally:
        driver.quit()

# Example usage
if __name__ == "__main__":
    result = fetch_fund_risk_statistics("FGPMX")
    print(json.dumps(result, indent=2))
