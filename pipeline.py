from stg1_ingest import IngestData
from stg2_trainAndevaluation import Evaluation
from stg3_train import Training
from stg4_predict import Prediction
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

        STAGE_NAME = "TRAIN&EVALUATION"
        logger.info(f"[{STAGE_NAME}]: Start training and evaluating data")
        init_eval = Evaluation()
        init_eval.eval(dir_path)
        init_eval.output_to_db()
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "FULL TRAINING"
        logger.info(f"[{STAGE_NAME}]: Start training full data")
        init_train = Training()
        init_train.main(dir_path, n_futures=30, epochs=10)
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")

        STAGE_NAME = "PREDICTION"
        logger.info(f"[{STAGE_NAME}]: Start prediction")
        init_prediction = Prediction()
        init_prediction.predict(dir_path)
        init_prediction.output_to_db()
        logger.info(f"[{STAGE_NAME}]: STAGE COMPLETE")




if __name__ == "__main__":
    # print("Hello World")
    dir_path = "./storage"
    init_pipeline = Pipeline()
    init_pipeline.main(dir_path)