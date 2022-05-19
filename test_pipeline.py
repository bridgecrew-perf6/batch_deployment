from classification_model.make_predictions import predict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
args = parser.parse_args()

data_path = args.data_path
predict(data_path)