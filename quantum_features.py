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
