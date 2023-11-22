import sys
from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
from src.ML_Project.components.data_ingestion import DataIngestion
from src.ML_Project.components.data_transformation import DataTransformationConfig, DataTransformation
from src.ML_Project.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model training
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(score)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)