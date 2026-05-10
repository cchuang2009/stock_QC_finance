import numpy as np
import yfinance as yf
import joblib
import os

from feature_engine import compute_features
from quantum_features import QuantumFeatureExtractor
from model import AlphaModel
from price_model import PricePredictor
from signal_engine import SignalEngine

MODEL_PATH = "alpha_model.pkl"

class HybridEngine:

    def __init__(self):

        self.qfe = QuantumFeatureExtractor(n_qubits=4)

        self.model = AlphaModel()

        self.price_model = PricePredictor()

        self.signal_engine = SignalEngine()

    def load_data(self, ticker):

        df = yf.Ticker(ticker).history(
            period="6mo",
            interval="1h",
            auto_adjust=True,
            actions=False
        )

        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")

        df = df.dropna()

        df = df[~df.index.duplicated()]

        return df

    def build_dataset(self, ticker):

        df = self.load_data(ticker)

        df = compute_features(df)

        base_cols = [
            "ret",
            "vol_z",
            "trend",
            "accel",
            "range"
        ]

        X_classical = df[base_cols].values

        q_features = []

        for x in X_classical:
            q_features.append(
                self.qfe.extract_features(x[:4])
            )

        q_features = np.array(q_features)

        X = np.hstack([X_classical, q_features])

        y = (
            df["Close"].shift(-5) > df["Close"]
        ).astype(int).values[:-1]

        return X[:-1], y

    def train(self, ticker="AAOI"):

        df = self.load_data(ticker)

        df = compute_features(df)

        base_cols = [
           "ret",
           "vol_z",
           "trend",
           "accel",
           "range"
        ]

        X_classical = df[base_cols].values

        q_features = []

        for x in X_classical:
            q_features.append(
               self.qfe.extract_features(x[:4])
            )

        q_features = np.array(q_features)

        X = np.hstack([X_classical, q_features])

        # --- targets ---
        y_class = (
            df["Close"].shift(-5) > df["Close"]
        ).astype(int)

        y_price = df["Close"].shift(-5)

        # --- combine and clean ---
        target_df = df.copy()

        target_df["y_class"] = y_class
        target_df["y_price"] = y_price

        target_df = target_df.dropna()

        valid_len = len(target_df)

        X = X[:valid_len]

        y_class = target_df["y_class"].values

        y_price = target_df["y_price"].values

        # --- train ---
        self.model.train(X, y_class)

        self.price_model.train(X, y_price)

    def predict(self, ticker="AAOI"):

        if not getattr(self.model, "fitted", False):
            self.train(ticker)

        df = self.load_data(ticker)

        df = compute_features(df)
        df = df.dropna()

        base_cols = [
            "ret",
            "vol_z",
            "trend",
            "accel",
            "range"
        ]

        X_classical = df[base_cols].values

        q_features = []

        for x in X_classical:
            q_features.append(
                self.qfe.extract_features(x[:4])
            )

        q_features = np.array(q_features)

        X = np.hstack([X_classical, q_features])

        X_recent = X[-20:]

        probs = self.model.predict(X_recent)

        price_preds = self.price_model.predict(X_recent)

        signal_df = self.signal_engine.generate(df, probs)

        signal_df["pred_price"] = price_preds

        signal_df["current_price"] = df["Close"].values[-20:]

        signal_df["expected_return"] = (
            signal_df["pred_price"]
            - signal_df["current_price"]
        ) / signal_df["current_price"]

        return signal_df
