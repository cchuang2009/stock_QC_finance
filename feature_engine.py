import numpy as np
import pandas as pd

def compute_features(df):
    """
    df: OHLCV dataframe
    """

    df = df.copy()

    df["ret"] = df["Close"].pct_change()
    df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    df["trend"] = df["Close"].rolling(10).mean() - df["Close"].rolling(30).mean()

    df["accel"] = df["ret"].diff()

    df["range"] = df["High"] - df["Low"]

    df = df.dropna()

    return df
