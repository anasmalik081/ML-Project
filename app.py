import sys
from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)