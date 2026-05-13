class SignalEngine:

    def generate(self, df, probs):

        df = df.iloc[-len(probs):].copy()

        df["prob"] = probs

        signals = []

        strengths = []

        for _, row in df.iterrows():

            prob = row["prob"]

            vol = row["vol_z"]

            trend = row["trend"]

            accel = row["accel"]

            # STRONG BUY
            if prob > 0.68 and vol > 1.5 and trend > 0:

                signal = "STRONG BUY"

            # BUY
            elif prob > 0.58 and trend > 0 and accel > 0:

                signal = "BUY"

            # STRONG SELL
            elif prob < 0.32 and vol > 1.5:

                signal = "STRONG SELL"

            # SELL
            elif prob < 0.42 and trend < 0:

                signal = "SELL"

            else:

                signal = "HOLD"

            strength = (
                prob
                * abs(vol)
                * (1 + abs(trend))
            )

            signals.append(signal)

            strengths.append(strength)

        df["signal"] = signals

        df["strength"] = strengths

        return df[
            [
                "prob",
                "signal",
                "strength"
            ]
        ]
