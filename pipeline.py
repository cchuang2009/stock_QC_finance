import numpy as np
import pandas as pd
import yfinance as yf

from feature_engine import compute_features
from quantum_features import QuantumFeatureExtractor
from model import AlphaModel
from price_model import PricePredictor
from linear_model_engine import LinearPriceModel
from signal_engine import SignalEngine


class HybridEngine:

    def __init__(self):

        self.qfe = QuantumFeatureExtractor(n_qubits=4)

        self.model = AlphaModel()

        self.price_model = PricePredictor()

        self.linear_model = LinearPriceModel()

        self.signal_engine = SignalEngine()

        self.feature_names = None

    # ---------------------------------------------------
    # Load Data
    # ---------------------------------------------------

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

    # ---------------------------------------------------
    # Feature Builder
    # ---------------------------------------------------

    def build_features(self, df):

        base_cols = [
            "ret",
            "vol_z",
            "trend",
            "accel",
            "range"
        ]

        X_classical = df[base_cols].copy()

        q_features = []

        for x in X_classical.values:

            q_features.append(
                self.qfe.extract_features(x[:4])
            )

        q_features = np.array(q_features)

        q_df = pd.DataFrame(
            q_features,
            columns=[
                "q_mean",
                "q_std",
                "q_entropy"
            ],
            index=X_classical.index
        )

        X = pd.concat(
            [X_classical, q_df],
            axis=1
        )

        X = X.fillna(0)

        return X

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------

    def train(self, ticker="AAOI"):

        df = self.load_data(ticker)

        df = compute_features(df)

        df = df.dropna()

        X = self.build_features(df)

        # future return target
        future_return = (
            df["Close"].shift(-5) - df["Close"]
        ) / df["Close"]

        # classification target
        y_class = (
            future_return > 0.03
        ).astype(int)

        # regression target
        y_price = future_return

        target_df = df.copy()

        target_df["y_class"] = y_class

        target_df["y_price"] = y_price

        target_df = target_df.dropna()

        valid_len = len(target_df)

        X = X.iloc[:valid_len]

        y_class = target_df["y_class"].values

        y_price = target_df["y_price"].values

        X = X.fillna(0)

        self.feature_names = X.columns.tolist()

        # train models
        self.model.train(X, y_class)

        self.price_model.train(X, y_price)

        self.linear_model.train(X, y_price)

    # ---------------------------------------------------
    # Predict
    # ---------------------------------------------------

    def predict(self, ticker="AAOI"):

        if not getattr(self.model, "fitted", False):
            self.train(ticker)

        df = self.load_data(ticker)

        df = compute_features(df)

        df = df.dropna()

        X = self.build_features(df)

        # enforce same feature order
        X = X[self.feature_names]

        X_recent = X.iloc[-20:]

        X_recent = X_recent.fillna(0)

        # probability prediction
        probs = self.model.predict(X_recent)

        # price prediction
        pred_lgb = self.price_model.predict(X_recent)

        pred_linear = self.linear_model.predict(X_recent)

        # ensemble
        price_preds = (
            0.7 * pred_lgb
            + 0.3 * pred_linear
        )

        # clamp insane predictions
        price_preds = np.clip(
            price_preds,
            -0.15,
            0.15
        )

        signal_df = self.signal_engine.generate(
            df,
            probs
        )

        current_prices = df["Close"].values[-20:]

        signal_df["current_price"] = current_prices

        signal_df["pred_return"] = price_preds

        # convert return -> price
        signal_df["pred_price"] = (
            current_prices * (1 + price_preds)
        )

        signal_df["expected_return"] = (
            signal_df["pred_price"]
            - signal_df["current_price"]
        ) / signal_df["current_price"]

        # confidence
        signal_df["confidence"] = (
            abs(signal_df["prob"] - 0.5) * 2
        )

        return signal_df
