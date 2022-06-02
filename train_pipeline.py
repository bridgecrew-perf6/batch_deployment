from classification_model.train_pipeline import train
from classification_model.config.paths_config import TRAINING_DATA_PATH
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_data_path", type=str, default=TRAINING_DATA_PATH)
args = parser.parse_args()

training_data_path = args.training_data_path
train(training_data_path)