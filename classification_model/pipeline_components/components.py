import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna("missing")

        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self):
        pass 
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.get_dummies(X, drop_first=True)

        return X

class Scaler(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X):
        X = X.copy()
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.variables].astype(float))
        
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = self.scaler.transform(X[self.variables].astype(float))

        return X

class Classifier(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self) -> None:
        pass
    def fit(self, X):
        X = X.copy()
        y = X.iloc[:, -1]
        self.classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        self.classifier.fit(X, y)
        
        return self

    def transform(self, X):
        y_pred = self.classifier.predict(X)

        return y_pred
    
    def predict(self, X):
        y_pred = self.transform(X)

        return y_pred