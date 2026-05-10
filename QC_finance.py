import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    class QuantumFeatureExtractor:
        """
        Encode classical features into quantum state and extract observables
        """

        def __init__(self, n_qubits=4):
            self.n_qubits = n_qubits

        def encode(self, x):
            """
            Angle encoding (stable for finance data)
            """
            qc = QuantumCircuit(self.n_qubits)

            x = np.tanh(x)  # stabilize

            for i in range(self.n_qubits):
                qc.ry(x[i % len(x)] * np.pi, i)

            # entangle
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)

            return qc

        def extract_features(self, x):
            qc = self.encode(x)
            state = Statevector.from_instruction(qc)

            probs = state.probabilities()

            # quantum-derived features
            qf1 = np.mean(probs)
            qf2 = np.std(probs)
            qf3 = entropy(probs)

            return np.array([qf1, qf2, qf3])


    def entropy(p):
        p = np.clip(p, 1e-12, 1)
        return -np.sum(p * np.log(p))

    return


@app.cell
def _():
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

    return


if __name__ == "__main__":
    app.run()
