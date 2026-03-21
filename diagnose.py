import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Referer': 'https://www.screener.in/',
})

r = session.get('https://www.screener.in/company/TCS/consolidated/', timeout=20)
print(f"Status: {r.status_code}")
print(f"URL: {r.url}")
print(f"Content length: {len(r.text)}")
print(f"\nFirst 1000 chars:")
print(r.text[:1000])

soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table')
print(f"\nTables found: {len(tables)}")
for i, t in enumerate(tables[:3]):
    print(f"Table {i}: {str(t)[:200]}")