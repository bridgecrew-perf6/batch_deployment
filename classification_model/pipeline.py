from classification_model.pipeline_components.components import *
from classification_model.config import config

from sklearn.pipeline import Pipeline

classification_pipeline = Pipeline(
    [
        (
            "categorical_imputer",
            CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "categorical_encoder",
            CategoricalEncoder(),
        ),
        (
            "scaler",
            Scaler(variables=config.NUMERICAL_VARS),
        ),
        (
            "classifier",
            Classifier(),
        ),
    ]
)