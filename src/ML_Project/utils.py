import os
import sys
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            model = model
            para = params[name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # training model with best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # predicting
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)