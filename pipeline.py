import numpy as np
import yfinance as yf

from feature_engine import compute_features
from quantum_features import QuantumFeatureExtractor
from model import AlphaModel
from signal_engine import SignalEngine
import joblib
import os

MODEL_PATH = "alpha_model.pkl"

def train(self, ticker="AAOI"):
    X, y = self.build_dataset(ticker)
    self.model.train(X, y)

    joblib.dump(self.model, MODEL_PATH)


def load_model(self):
    if os.path.exists(MODEL_PATH):
        self.model = joblib.load(MODEL_PATH)


class HybridEngine:

    def __init__(self):
        self.qfe = QuantumFeatureExtractor(n_qubits=4)
        self.model = AlphaModel()
        self.signal_engine = SignalEngine()
    # ✅ ADD THIS FUNCTION
    def load_data(self, ticker):
        df = yf.Ticker(ticker).history(
            period="6mo",
            interval="1h",
            auto_adjust=True,
            actions=False
        )

        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")

        # clean data (important)
        df = df.dropna()
        df = df[~df.index.duplicated()]

        return df

    def build_dataset(self, ticker):
        df = yf.download(ticker, period="6mo", interval="1h")
        df = compute_features(df)

        X_classical = df[["ret", "vol_z", "trend", "accel", "range"]].values

        q_features = []
        for x in X_classical:
            q_features.append(self.qfe.extract_features(x[:4]))

        q_features = np.array(q_features)

        X = np.hstack([X_classical, q_features])

        y = (df["Close"].shift(-1) > df["Close"]).astype(int).values[:-1]

        return X[:-1], y

    def train(self, ticker="AAOI"):
        X, y = self.build_dataset(ticker)
        self.model.train(X, y)

    def predict(self, ticker="AAOI"):

        if not getattr(self.model, "fitted", False):
           self.train(ticker)

        df = self.load_data(ticker)
        df = compute_features(df)

        base_cols = ["ret", "vol_z", "trend", "accel", "range"]
        X_classical = df[base_cols].values

        q_features = []
        for x in X_classical:
            q_features.append(self.qfe.extract_features(x[:4]))

        q_features = np.array(q_features)

        X = np.hstack([X_classical, q_features])

        probs = self.model.predict(X[-20:])

        signal_df = self.signal_engine.generate(df, probs)

        return signal_df
