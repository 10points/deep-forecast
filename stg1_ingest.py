from utils.utils import DataIngestion
import os
from dotenv import load_dotenv

class IngestData:
    def __init__(self):
        pass

    def fetch(self):
        # Import data
        load_dotenv()
        user_name = os.getenv("USER_NAME")
        password = os.getenv("PASSWORD")
        ip = os.getenv("IP_ADDRESS")
        port = os.getenv("PORT")
        db_name = os.getenv("DB_NAME")
        storage_path = os.getenv("STORAGE_PATH")

        get_data = DataIngestion(
            user_name=user_name,
            password=password,
            ip=ip,
            port=port,
            DB_name=db_name,
            path=storage_path
        )
        get_data.IngestionContinuousPrice()
        get_data.IngestContractPrice()

if __name__ == "__main__":
    init_ingest = IngestData()
    init_ingest.fetch()



