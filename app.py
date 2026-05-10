import streamlit as st
from pipeline import HybridEngine

st.title("⚛️ Quantum + ML Alpha Engine")

engine = HybridEngine()

ticker = st.text_input("Ticker", "AAOI")

if st.button("Train Model"):
    engine.train(ticker)
    st.success("Model trained")

if st.button("Run Prediction"):
    try:
        result = engine.predict(ticker)

        st.subheader("📊 Trading Signals")
        st.dataframe(result)

        # show latest signal
        latest = result.iloc[-1]

        st.metric("Signal", latest["signal"])
        st.metric("Probability", round(latest["prob"], 3))
        st.metric("Strength", round(latest["strength"], 3))

    except Exception as e:
        st.error(f"Error: {e}")
