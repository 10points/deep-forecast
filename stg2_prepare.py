import pandas as pd
import numpy as np
from datetime import date
from utils.utils import get_contract_price_by_market_month
from dotenv import load_dotenv
import os
from logger import logger


STAGE_NAME = "DATA PREPARATION"
month_mapping = {"JAN":"01","MAR":"03","MAY":"05","JLY":"07","AUG":"08","SEP":"09","NOV":"11"}

class DataPreparation:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def get_contract_price(self):
        file = "contract_data.csv"
        self.get_contract_price = pd.read_csv(os.path.join(self.dir_path,file) , parse_dates=["data_date"])
        return self.get_contract_price

    def prepare_data_to_train(self):
        logger.info(f"[{STAGE_NAME}]: start prepare contract data")
        contract_data =  self.get_contract_price()  
        # print("contract_data: ",contract_data.head())
        current_date = date.today()
        print(f"current_date: {current_date}")
        market_month_to_train = {"market_month":[], "expired_date":[], "max_contrat_date":[]}

        for contract in contract_data["market_month"].unique():
            df = get_contract_price_by_market_month(self.dir_path, contract)
            # print("by contract",df.head())
            max_contract_data_date = df["data_date"].max()
            # print(f"max_contract_data_date {max_contract_data_date}")
            diff = np.busday_count(max_contract_data_date.date(),current_date)
            # print(f"contract:{contract} -- max_date:{max_contract_data_date.date()} cuuent:{current_date} -- diff:{diff}") 
            # print(f"{contract} : {df.market_month.count()}")
            if diff <= 2 and df.shape[0] >= 320:
                print("--------Entered condition--------")
                temp1, temp2 = contract.split(" ")
                month = month_mapping[temp1]
                year = "20" + temp2
                market_month_to_train["market_month"].append(contract)
                expire_date = year+"-"+month+"-14"
                market_month_to_train["expired_date"].append(expire_date)
                market_month_to_train["max_contrat_date"].append(max_contract_data_date)


        output = pd.DataFrame(market_month_to_train)
        output["expired_date"] = pd.to_datetime(output["expired_date"])
        output["expired_date"] = output["expired_date"].apply(lambda x: x-pd.DateOffset(x.day_of_week-4) if x.day_of_week - 4 >= 1 else x)
        output["max_pred_day"] =  output["expired_date"].apply(lambda x : int(np.busday_count(market_month_to_train["max_contrat_date"][0].date(), x.date())))
        # output.to_csv(Path(f"./Data_Preparation/market_month_to_train_{datetime.now()}.csv"), index=False)
        file_name = "market_month_to_train.csv"
        output.to_csv(os.path.join(self.dir_path, file_name), index=False)
        logger.info(f"[{STAGE_NAME}]: Prepare contract data completed")

if __name__ == "__main__":
    load_dotenv()
    storage_path = os.getenv("STORAGE_PATH")
    # storage_path = "../storage"
    init_prepare = DataPreparation(dir_path=storage_path)
    init_prepare.prepare_data_to_train() 