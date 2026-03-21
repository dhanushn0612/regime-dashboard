
import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.screener.in/",
})

r = session.get("https://www.screener.in/company/TCS/consolidated/", timeout=20)
soup = BeautifulSoup(r.text, "html.parser")

# Show all table headers and first data row
tables = soup.find_all("table", class_="data-table")
print(f"data-table tables found: {len(tables)}")

for i, table in enumerate(tables[:6]):
    # Get headers
    headers = []
    for th in table.find_all("th"):
        txt = th.get_text(strip=True)
        if txt:
            headers.append(txt[:15])
    print(f"\nTable {i} headers: {headers[:8]}")
    
    # Get first 3 data rows
    for j, row in enumerate(table.find_all("tr")[1:4]):
        cells = row.find_all("td")
        if cells:
            row_data = [c.get_text(strip=True)[:12] for c in cells[:6]]
            print(f"  Row {j}: {row_data}")

# Also check the ratios section specifically
ratios = soup.find("section", id="ratios")
if ratios:
    print("\nRatios section found")
    table = ratios.find("table")
    if table:
        for row in table.find_all("tr")[:5]:
            cells = row.find_all("td")
            if cells:
                print(f"  {[c.get_text(strip=True)[:15] for c in cells[:5]]}")
else:
    print("\nNo ratios section found - checking all section IDs:")
    for s in soup.find_all("section"):
        print(f"  id={s.get('id','none')} class={s.get('class','none')}")
