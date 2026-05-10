import numpy as np

class SimpleQAOASelector:
    """
    Lightweight classical simulation of QAOA-like selection
    """

    def select(self, scores, k=3):
        """
        Select top-k "quantum-enhanced" assets
        """
        idx = np.argsort(scores)[::-1]
        return idx[:k]
