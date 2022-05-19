from classification_model.pipeline import classification_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

import logging
import pickle

_logger = logging.getLogger(__name__)


def train(training_data_path: str) -> None:
    _logger.info(f"Reding the data from {training_data_path}")
    data_df = pd.read_csv(training_data_path)
    # train, test = train_test_split(data_df, test_size=0.2)
    train = data_df
    test = data_df

    _logger.info("Training the model")
    print(train.head(2))
    print(test.head(2))
    y_train_pred = classification_pipeline.fit_transform(train)
    y_train = pd.get_dummies(train.iloc[:, -1], drop_first=True).values
    train_acc = (y_train == y_train_pred).mean()

    y_test_pred = classification_pipeline.predict(test)
    y_test = pd.get_dummies(test.iloc[:, -1], drop_first=True).values
    test_acc = (y_test == y_test_pred).mean()

    _logger.info(f"Train accuracy is: {train_acc}. Test accuracy is {test_acc}")
    _logger.info("Saving the model")
    pickle.dump(classification_pipeline, open("saved_models/model.pkl", "wb"))