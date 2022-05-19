from classification_model.pipeline import classification_pipeline
import pandas as pd
import pickle

import logging
_logger = logging.getLogger(__name__)

def predict(data_path: str) -> None:
    _logger.info(f"Reding the data from {data_path}")
    data_df = pd.read_csv(data_path)
    _logger.info("Loading the classification pipeline.")
    classification_pipeline = pickle.load(open("saved_models/model.pkl", "rb"))

    _logger.info("Making the predictions")
    y_test_pred = classification_pipeline.predict(data_df)

    print(y_test_pred)