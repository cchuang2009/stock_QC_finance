import pandas as pd
from pipeline import HybridEngine

TICKERS = [
    "AAOI",
    "LITE",
    "COHR",
    "MRVL",
    "NVDA",
    "AMD",
    "MU",
    "CRDO",
    "POET",
    "LWLG"
]

engine = HybridEngine()

results = []

for ticker in TICKERS:

    try:

        df = engine.predict(ticker)

        latest = df.iloc[-1]

        results.append({
            "ticker": ticker,
            "signal": latest["signal"],
            "prob": latest["prob"],
            "pred_price": latest["pred_price"],
            "current_price": latest["current_price"],
            "expected_return": latest["expected_return"],
            "strength": latest["strength"]
        })

    except Exception as e:
        print(ticker, e)

scan_df = pd.DataFrame(results)

scan_df = scan_df.sort_values(
    by="strength",
    ascending=False
)

print(scan_df)