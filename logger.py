import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

LOGS_DIRECTORY = "./logs"
os.makedirs(LOGS_DIRECTORY, exist_ok=True)
Log_FILE_PATH = None

def create_logger(module_name):
    global Log_FILE_PATH

    if Log_FILE_PATH is None:
        Log_FILE_PATH = os.path.join(LOGS_DIRECTORY, f"logger_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
        
    logger = logging.getLogger(module_name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)

    file_handler = TimedRotatingFileHandler(Log_FILE_PATH, when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    file_handler.flush = lambda: file_handler.stream.flush()
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

                    