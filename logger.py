import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler


logging_str = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"

log_dir = "./logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout), # printing log inside terminal
        TimedRotatingFileHandler(filename=log_filepath, when='midnight', interval=1, backupCount=30, encoding='utf-8', delay=False)
    ]
)

logger = logging.getLogger(__name__)