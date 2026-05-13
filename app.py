import streamlit as st
import pandas as pd

from pipeline import HybridEngine

st.set_page_config(
    page_title="Quantum + ML Alpha Engine",
    layout="wide"
)

st.title("⚛️ Quantum + ML Alpha Engine")

# -------------------------
# Persistent Engine
# -------------------------

if "engine" not in st.session_state:
    st.session_state.engine = HybridEngine()

engine = st.session_state.engine

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input(
    "Ticker",
    "AAOI"
)

# -------------------------
# Main Prediction
# -------------------------

st.header(f"📈 Single Stock Prediction: {ticker}")

if st.button("Run Prediction"):

    try:

        result = engine.predict(ticker)

        latest = result.iloc[-1]

        # -------------------------
        # Metrics
        # -------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Signal",
            latest["signal"]
        )

        col2.metric(
            "Probability",
            round(latest["prob"], 3)
        )

        col3.metric(
            "Expected Return",
            f"{latest['expected_return']*100:.2f}%"
        )

        col4.metric(
            "Confidence",
            round(latest["confidence"], 3)
        )

        # -------------------------
        # Price Information
        # -------------------------

        st.subheader("💰 Price Prediction")

        price_df = pd.DataFrame({
            "Current Price": result["current_price"],
            "Predicted Price": result["pred_price"],
            "Predicted Return": result["pred_return"],
            "Expected Return": result["expected_return"]
        })

        st.dataframe(price_df)

        # -------------------------
        # Signal Table
        # -------------------------

        st.subheader("📊 Signal History")

        st.dataframe(result)

    except Exception as e:

        st.error(f"Error: {e}")

# -------------------------
# Multi-Stock Scanner
# -------------------------

st.header("🚀 Multi-Stock Quantum Scanner")

TICKERS = [
    "AAOI",
    "aehr",
    "AG",
    "AGIX",
    "AI",
    "alab",
    "AMAT",
    "AMD",
    "anet",
    "BBAI",
    "BLSH",
    "BTG",
    "CCOI",
    "COHR",
    "CRDO",
    "DGXX",
    "DXYZ",
    "INTC",
    "IONQ",
    "ISRG",
    "LITE",
    "LMT",
    "NOK",
    "NUAI",
    "MELI",
    "MRVL",
    "MU",
    "NBIS",
    "NVDA",
    "PLTR",
    "PAPL",
    "PLUG",
    "POET",
    "PYPL",
    "QBTS",
    "LWLG",
    "QUBT",
    "RGTI",
    "SMCI",
    "SNDK"
    ""
]

if st.button("Run Multi-Scanner"):

    scan_results = []

    progress = st.progress(0)

    for i, t in enumerate(TICKERS):

        try:

            df = engine.predict(t)

            latest = df.iloc[-1]

            scan_results.append({

                "Ticker": t,

                "Signal": latest["signal"],

                "Probability": round(
                    latest["prob"], 3
                ),

                "Current Price": round(
                    latest["current_price"], 2
                ),

                "Predicted Price": round(
                    latest["pred_price"], 2
                ),

                "Expected Return %": round(
                    latest["expected_return"] * 100,
                    2
                ),

                "Confidence": round(
                    latest["confidence"], 3
                ),

                "Strength": round(
                    latest["strength"], 3
                )
            })

        except Exception as e:

            scan_results.append({

                "Ticker": t,
                "Signal": "ERROR",
                "Probability": 0,
                "Current Price": 0,
                "Predicted Price": 0,
                "Expected Return %": 0,
                "Confidence": 0,
                "Strength": 0
            })

        progress.progress((i + 1) / len(TICKERS))

    scan_df = pd.DataFrame(scan_results)

    scan_df = scan_df.sort_values(
        by="Strength",
        ascending=False
    )

    # -------------------------
    # Top Opportunities
    # -------------------------

    st.subheader("🔥 Ranked Opportunities")

    st.dataframe(scan_df)

    # -------------------------
    # Best Signal Highlight
    # -------------------------

    top = scan_df.iloc[0]

    st.success(
        f"Top Opportunity: "
        f"{top['Ticker']} | "
        f"{top['Signal']} | "
        f"Expected Return: "
        f"{top['Expected Return %']}%"
    )
