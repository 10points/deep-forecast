from stg1_ingest import IngestData
from stg2_prepare import DataPreparation
from stg3_trainAndevaluation import Evaluation
from stg4_train import Training
from stg5_predict import Prediction
from dotenv import load_dotenv
import os
from logger import logger

class Pipeline:
    def __init__(self):
        pass

    def main(self, dir_path):

        STAGE_NAME = "INGEST DATA"
        logger.info(f"[{STAGE_NAME}]: Start ingesting data from DB")
        init_ingest = IngestData()
        init_ingest.fetch()
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "PREPARE DATA"
        logger.info(f"[{STAGE_NAME}]: Start prepare data market_month contract")
        init_prepare = DataPreparation(dir_path)
        init_prepare.prepare_data_to_train() 
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "TRAIN&EVALUATION"
        logger.info(f"[{STAGE_NAME}]: Start training and evaluating data")
        init_eval = Evaluation()
        init_eval.eval(dir_path, epochs=300)
        init_eval.output_to_db()
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "FULL TRAINING"
        logger.info(f"[{STAGE_NAME}]: Start training full data")
        init_train = Training()
        init_train.main(dir_path, n_futures=30, epochs=300)
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "PREDICTION"
        logger.info(f"[{STAGE_NAME}]: Start prediction")
        init_prediction = Prediction()
        init_prediction.predict(dir_path)
        init_prediction.output_to_db()
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")




if __name__ == "__main__":
    # print("Hello World")
    load_dotenv()
    dir_path= os.getenv("STORAGE_PATH")
    # dir_path = "./storage"
    init_pipeline = Pipeline()
    init_pipeline.main(dir_path)