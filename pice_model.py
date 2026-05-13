import lightgbm as lgb


class PricePredictor:

    def __init__(self):

        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8
        )

        self.fitted = False

    def train(self, X, y):

        self.model.fit(X, y)

        self.fitted = True

    def predict(self, X):

        if not self.fitted:
            raise ValueError("Price model not trained")

        return self.model.predict(X)
