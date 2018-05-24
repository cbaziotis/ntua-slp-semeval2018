from sklearn.base import BaseEstimator, TransformerMixin


class Untokenizer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return [" ".join(x) for x in X]

    def fit(self, X, y=None):
        return self
