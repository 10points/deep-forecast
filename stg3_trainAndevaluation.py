import pandas as pd
import numpy as np
from neuralforecast.models import LSTM, DilatedRNN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_squared_error
from utils.utils import ingest_exogenous_data, get_contract_price_by_market_month, preprocess, get_exo_columns, assign_rmse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import os
from logger import logger



class Evaluation:
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
        backtest_table = os.getenv("TABLE_BACKTEST")

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
        del_sql_sql = "DELETE FROM stg_forecast_soybean_backtest where model='{model}' AND contract_name='{contract_name}' AND data_date>='{data_date}'"

        for idx, row in result.iterrows():
            del_sql = del_sql_sql.format(model=idx[0], contract_name=idx[1], data_date=row['data_date'])
            print("######################")
            print(del_sql)
            print("######################")
            conn.execute(text(del_sql))
            conn.commit()


        self.df_all.to_sql(
            backtest_table, 
            engine, 
            if_exists='append', 
            index=False, 
            method="multi")
        conn.close()
        logger.info("Prediction data have been uploaded")

    def eval(self, dir_path):
        file_mkt_month = "market_month_to_train.csv"
        market_month_to_train = pd.read_csv(os.path.join(dir_path, file_mkt_month))
        #   storage_path = "./storage"
        exo_data = ingest_exogenous_data(dir_path)
        # print(exo_data)
        forecast_data = []
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
            actual_price = cleaned_contract["price_settle"] # non-smoothing price
            prep_data = preprocess(cleaned_contract)
            exo_col = get_exo_columns(prep_data)
            # break

            # train data
            back_test_period = [7, 15, 30]
            for num in back_test_period:
                logger.info(f"# of days in backtest: {num} days")
                trainsize = len(prep_data)-num
                train_df = prep_data[:trainsize]
                test_df = prep_data[-num:]
                price_settle = actual_price[-num:]

                horizon = len(test_df)

                models = [

                    LSTM(h = horizon,
                        input_size = -1,
                        encoder_n_layers=2,
                        encoder_hidden_size=128,
                        context_size=10,
                        decoder_hidden_size=128,
                        decoder_layers=2,
                        hist_exog_list = exo_col, # <- Historical exogenous variables
                        scaler_type = 'robust',
                        # dropout_prob_theta=0.5,
                        max_steps=10,
                        # early_stop_patience_steps=5,
                        loss=MAE()),
                    DilatedRNN(h = horizon,
                        input_size = -1,
                        encoder_hidden_size=128,
                        hist_exog_list = exo_col, # <- Historical exogenous variables
                        scaler_type = 'robust',
                        # dropout_prob_theta=0.5,
                        max_steps=10,
                        # early_stop_patience_steps=5,
                        loss=MAE())
                    ]
                nf = NeuralForecast(models=models, freq='D')
                nf.fit(df=train_df)

                # Evaluation
                Y_hat_df = nf.predict().reset_index() # model prediction

                y_hat = Y_hat_df.copy()
                y_hat = y_hat.drop(columns=["unique_id", "ds"])
                melted_df = pd.melt(y_hat, var_name='model', value_name='forecast_price')
                # actual = test_df["y"].reset_index().drop(columns="index")
                extended_array = np.concatenate((price_settle.values, price_settle.values), axis=0) # because there are 2 models and we have to duplicate test data for 2 times to compare alongside of dataframe
                melted_df["price_settle"] = extended_array
                melted_df["type"] = num
                melted_df["contract"] = market_month

                # add rmse metrics
                rmse_lstm = mean_squared_error(price_settle.values, Y_hat_df["LSTM"], squared=False)
                rmse_dilatedrnn = mean_squared_error(price_settle.values, Y_hat_df["DilatedRNN"], squared=False)
                melted_df["rmse"] = melted_df["model"].apply(lambda x: assign_rmse(x, rmse_lstm, rmse_dilatedrnn))

                # generate date index
                max_date = max(train_df["ds"])
                gen_date_index = pd.bdate_range(start=max_date + pd.Timedelta(1, unit="D"), periods=num, freq='B')
                gen_date_series = pd.Series(gen_date_index)
                concatdate = np.concatenate([gen_date_series.values, gen_date_series.values], axis=0)
                melted_df = melted_df.set_index(concatdate)
                melted_df = melted_df.rename_axis("data_date")

            
            
            # melted_df[["price_settle", "prediction", "model"]]


                forecast_data.append(melted_df)


        self.df_all = pd.concat(forecast_data, axis=0)
        self.df_all = self.df_all.rename(columns={"prediction":"forecast_price"})
        self.df_all = self.df_all.reset_index().rename(columns={"index":"data_date"})

        self.df_all[["data_date","price_settle", "forecast_price", "type", "model", "contract", "rmse"]]


        return self.df_all