from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json

def fetch_fund_risk_statistics(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/risk"

    # Chrome options
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--log-level=3")
    options.add_argument("user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        print(f"Loading {url} ...")
        driver.get(url)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")
        tables = soup.find_all("table")

        target_table = None
        for table in tables:
            headers = [th.text.strip() for th in table.find_all("th")]
            if "3 Years" in headers and "5 Years" in headers and "10 Years" in headers:
                target_table = table
                break

        if not target_table:
            print("Risk Statistics table not found.")
            return {}

        metrics = {}
        rows = target_table.find_all("tr")
        for row in rows[1:]:  
            cols = row.find_all("td")
            if len(cols) >= 6:  
                metric = cols[0].text.strip()
                metrics[metric] = {
                    "3y": cols[1].text.strip(),
                    "5y": cols[3].text.strip(),
                    "10y": cols[5].text.strip(),
                }

        return metrics

    finally:
        driver.quit()

if __name__ == "__main__":
    result = fetch_fund_risk_statistics("FGPMX")
    print(json.dumps(result, indent=2))
