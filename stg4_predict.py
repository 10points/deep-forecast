import pandas as pd
import numpy as np
from neuralforecast.core import NeuralForecast
from utils.utils import ingest_exogenous_data, get_contract_price_by_market_month, preprocess
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import os
from logger import logger

class Prediction:
  def __init__(self):
    pass

  def output_to_db(self):
 
    print("Push data to db")
    load_dotenv()
    drivername = os.getenv("DRIVER_NAME")
    username = os.getenv("USER_DB")
    host = os.getenv("HOST")
    port = os.getenv("PORT_DB")
    database= os.getenv("DATABASE")
    password = os.getenv("PASSWORD_DB")
    predict_table = os.getenv("TABLE_PREDICT")

    url = URL.create(
        drivername=drivername,
        username=username,
        host=host,
        port=port,
        database=database,
        password=password
    )
    engine = create_engine(url)
    conn = engine.connect()

    # group by contract name
    group = self.df_all.groupby(['model', 'contract_name'])
    # find max/min date for each contract name
    result = group.agg({'data_date':np.min})

    # query to DELETE every new prediction when getting latest prediction
    del_sql_sql = "DELETE FROM stg_forecast_soybean_price where model='{model}' AND contract_name='{contract_name}' AND data_date>='{data_date}'"

    for idx, row in result.iterrows():
        del_sql = del_sql_sql.format(model=idx[0], contract_name=idx[1], data_date=row['data_date'])
        print("######################")
        print(del_sql)
        print("######################")
        conn.execute(text(del_sql))
        conn.commit()


    self.df_all.to_sql(
        predict_table, 
        engine, 
        if_exists='append', 
        index=False, 
        method="multi")
    conn.close()
    logger.info("Prediction data have been uploaded")

  def predict(self, dir_path):

    forecast_df = []

    file_mkt_month = "market_month_to_train.csv"
    market_month_to_train = pd.read_csv(os.path.join(dir_path, file_mkt_month))
    
    #   storage_path = "./storage"
    exo_data = ingest_exogenous_data(dir_path)
    # print(exo_data)
    
    for market_month in market_month_to_train["market_month"]:
  
          
        contract_price = get_contract_price_by_market_month(dir_path=dir_path, market_month=market_month)
        contract_price = contract_price.drop_duplicates(subset=["data_date"])
        contract_price = contract_price.groupby("data_date")["price_settle"].sum().reset_index()
      

        if contract_price["price_settle"].isna().sum() > 0:
            contract_price.dropna(axis=0, inplace=True)

      
        # set index as date in contract price
        contract_price = contract_price.set_index("data_date")

        # all_data.append(contract_price)
        exo_data1 = exo_data.copy()
        exo_data1 = exo_data1.drop(columns="price_soybean-seed")
        
        cleaned_contract = pd.merge(exo_data1, contract_price, left_index=True, right_index=True)
        # print(cleaned_contract["price_settle"])
        prep_data = preprocess(cleaned_contract)        
        

        nf2 = NeuralForecast.load(path=f'./checkpoints/{market_month}/')
        Y_hat_df = nf2.predict().reset_index()

        # print(Y_hat_df.head())
        y_hat = Y_hat_df.copy()
        y_hat = y_hat.drop(columns=["unique_id", "ds"])
        melted_df = pd.melt(y_hat, var_name='model', value_name='forecast_price')
        melted_df["contract"] = market_month

        # generate date index
        max_date = max(prep_data["ds"])
        gen_date_index = pd.bdate_range(start=max_date + pd.Timedelta(1, unit="D"), periods=len(y_hat), freq='B')
        gen_date_series = pd.Series(gen_date_index)
        concatdate = np.concatenate([gen_date_series.values, gen_date_series.values], axis=0)
        melted_df = melted_df.set_index(concatdate)
        melted_df = melted_df.rename_axis("data_date")

        forecast_df.append(melted_df)

    self.df_all = pd.concat(forecast_df, axis=0)
    self.df_all = self.df_all.reset_index().rename(columns={"index":"data_date"})
    self.df_all = self.df_all[["data_date", "forecast_price", "model", "contract"]]

    return self.df_all