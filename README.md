# Ichimoku Weekly Scanner

Mobile-friendly scanner for the weekly Ichimoku kumo + Chikou breakout setup on NSE F&O stocks.

- `index.html` – the web app (static, reads `data/scan.json`), with Weekly and Monthly views
- `scanner/scan.py` – computes the scan from Yahoo Finance daily prices
- `scanner/universe.csv` – stock universe (NSE F&O list)
- `.github/workflows/scan.yml` – refreshes the data after each market close

Run locally: `pip install -r requirements.txt && python scanner/scan.py`, then serve the folder
(`python -m http.server`) and open it in a browser.

For research and education only. Not investment advice.
