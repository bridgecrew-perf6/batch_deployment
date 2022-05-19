from classification_model.train_pipeline import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_data_path", type=str)
args = parser.parse_args()

training_data_path = args.training_data_path
train(training_data_path)