import lightgbm as lgb

class PricePredictor:

    def __init__(self):

        self.model = lgb.LGBMRegressor(
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
            raise ValueError("Price model not trained")

        return self.model.predict(X)