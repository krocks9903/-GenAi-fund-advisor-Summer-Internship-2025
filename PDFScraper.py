import fitz  # PyMuPDF
import re
import pdfplumber
import pandas as pd

pdf_path = 'your_pdf_here.pdf'


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

def extract_patterns(text):
    patterns = {
        'fund_name': r'(?:Fund Name|Scotia Canadian Equity Fund - Series A)',
        'inception_date': r'(?:Date series started|Start Date)[^\d]*(\w+ \d{1,2},? \d{4})',
        'aum': r'(?:Total value of Fund on [^\:]*:|\bTotal Assets\s*\$mil)\s*\$?([\d,.]+[MB]?)',
        'mer': r'(?:Management expense ratio \(MER\)|MER \(as of[^\)]*\))\s*[:\s]*([\d\.]+%)',
        'risk_rating': r'Risk Rating\s*\n*\s*(\w+)',
        'benchmark': r'Benchmark\s*(.*?)\n',
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted[key] = match.group(1).strip() if match else None
    return extracted

def extract_top_holdings(text):
    # Common pattern for top holdings → either "Top 10 investments" or "Top 10 Holdings"
    holdings_pattern = r'(?:Top 10 investments|Top 10 Holdings)[\s\S]+?(\d+\.\d+%.*?)\n\n'
    match = re.search(holdings_pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    holdings_text = match.group(1)
    holdings = []
    for line in holdings_text.split('\n'):
        holding_match = re.match(r'^(.*?)(\d+\.\d+%)$', line.strip())
        if holding_match:
            company, percent = holding_match.groups()
            holdings.append({'company': company.strip(), 'percent': percent})
    return holdings

def extract_sector_allocation(text):
    # Search for Sector Allocation blocks
    sector_pattern = r'(?:Investment mix|Top 5 Sector Allocation)[\s\S]+?(\d+\.\d+%.*?)\n\n'
    match = re.search(sector_pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    sector_text = match.group(1)
    sectors = []
    for line in sector_text.split('\n'):
        sector_match = re.match(r'^(.*?)(\d+\.\d+%)$', line.strip())
        if sector_match:
            sector, percent = sector_match.groups()
            sectors.append({'sector': sector.strip(), 'percent': percent})
    return sectors

if __name__ == "__main__":
    # Extract text
    text = extract_text(pdf_path)
    print("---- First 1000 characters ----")
    print(text[:1000])
    
    # Extract key patterns
    data = extract_patterns(text)
    print("\n---- Key Fund Data ----")
    for k, v in data.items():
        print(f"{k}: {v}")
    
    # Extract top holdings
    top_holdings = extract_top_holdings(text)
    print("\n---- Top 10 Holdings ----")
    for holding in top_holdings:
        print(f"{holding['company']}: {holding['percent']}")
    
    # Extract sector allocation
    sector_alloc = extract_sector_allocation(text)
    print("\n---- Sector Allocation ----")
    for sector in sector_alloc:
        print(f"{sector['sector']}: {sector['percent']}")
    
    # Extract tables (optional - for year-by-year returns or compound returns)
    tables = extract_tables(pdf_path)
    print(f"\n---- Extracted {len(tables)} Tables ----")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}:")
        print(table.head())

