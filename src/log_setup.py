import warnings
warnings.filterwarnings("ignore")
import logging
import os
from datetime import datetime

def setup_logger(module_name, log_dir=None):
    logger = logging.getLogger(f"{module_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        if not log_dir:
            log_dir = "all_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%d-%m-%Y')}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    
    return logger
