FROM python:3.6

COPY . /home/credict_classifier/
WORKDIR /home/credict_classifier/

RUN pip install -r requirements.txt
RUN python train_pipeline.py --training_data_path data/german_credit_data.csv

CMD ["python", "test_pipeline.py", "--data_path", "data/german_credit_data.csv"]