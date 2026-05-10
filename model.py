import lightgbm as lgb
import numpy as np

class AlphaModel:

    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5
        )
        self.fitted = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

