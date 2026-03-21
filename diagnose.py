import os, pickle

script_dir = r"C:\Users\Dhanush\regime-dashboard\data_pipeline"
cache_path = os.path.join(script_dir, "nifty500_prices.pkl")

print(f"Cache path: {cache_path}")
print(f"Exists: {os.path.exists(cache_path)}")

if os.path.exists(cache_path):
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"Size: {size_mb:.1f} MB")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    print(f"Stocks in cache: {len(data)}")
    print(f"Sample tickers: {list(data.keys())[:5]}")
else:
    print("Cache NOT found - listing data_pipeline contents:")
    for f in os.listdir(script_dir):
        print(f"  {f}")