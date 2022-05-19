import os
import logging

from classification_model.config import logging_config

if not os.path.exists('saved_models'):
    os.mkdir('saved_models')
    
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False