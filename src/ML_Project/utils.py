import os
import sys
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        my_db = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established", my_db)
        df = pd.read_sql_query("SELECT * FROM students", my_db)
        print(df.head())
        return df
    except Exception as e:
        raise CustomException(e, sys)