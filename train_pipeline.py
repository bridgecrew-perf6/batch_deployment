from classification_model.train_pipeline import train
from classification_model.config.paths_config import training_data_path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_data_path", type=str, default=training_data_path)
args = parser.parse_args()

training_data_path = args.training_data_path
train(training_data_path)