from sklearn.linear_model import Ridge


class LinearPriceModel:

    def __init__(self):

        self.model = Ridge(alpha=1.0)

        self.fitted = False

    def train(self, X, y):

        self.model.fit(X, y)

        self.fitted = True

    def predict(self, X):

        if not self.fitted:
            raise ValueError("Linear model not trained")

        return self.model.predict(X)
