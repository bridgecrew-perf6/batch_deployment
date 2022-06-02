import boto3
import logging

from classification_model.config import aws_config

_logger = logging.getLogger(__name__)

def save_model_to_s3(model_pickle: object):
    """
    Function for writing the trained model pickle to s3
    """
    try:
        session = boto3.Session(
            region_name='us-east-2'
        )
        s3 = session.resource('s3')
        _logger.info("Writing the trained model to s3")
        s3.Object(aws_config.S3_BUCKET, aws_config.TRAINED_MODEL_PATH).put(Body=model_pickle)
        _logger.info("Succesfully saved model to s3.")
    except Exception as e:
        _logger.error(f"Unable to write the model to s3: {e}")