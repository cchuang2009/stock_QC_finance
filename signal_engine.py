import numpy as np
import pandas as pd

class SignalEngine:

    def generate(self, df, probs):
        """
        df: feature dataframe (aligned)
        probs: model output
        """

        df = df.iloc[-len(probs):].copy()
        df["prob"] = probs

        signals = []
        strengths = []

        for _, row in df.iterrows():

            prob = row["prob"]
            vol = row["vol_z"]
            trend = row["trend"]
            accel = row["accel"]

            # --- BUY ---
            if prob > 0.7 and vol > 2 and trend > 0:
                signal = "BUY"

            # --- SELL ---
            elif prob < 0.4 and vol > 2 and accel < 0:
                signal = "SELL"

            else:
                signal = "HOLD"

            strength = prob * abs(vol) * (1 + abs(trend))

            signals.append(signal)
            strengths.append(strength)

        df["signal"] = signals
        df["strength"] = strengths

        return df[["prob", "signal", "strength"]]
