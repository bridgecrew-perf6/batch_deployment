FROM python:3.6

COPY . /home/credict_classifier/
WORKDIR /home/credict_classifier/

RUN pip install -r requirements.txt

CMD ["python", "train_pipeline.py"]