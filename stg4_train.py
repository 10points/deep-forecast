import pandas as pd
from neuralforecast.models import LSTM, DilatedRNN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from utils.utils import ingest_exogenous_data, get_contract_price_by_market_month, preprocess, get_exo_columns
import os

class Training:
  def __init__(self):
    pass


  def main(self, dir_path, n_futures: int, epochs: int):

      file_mkt_month = "market_month_to_train.csv"
      market_month_to_train = pd.read_csv(os.path.join(dir_path, file_mkt_month))
      market_month_to_train = market_month_to_train.set_index("market_month")
      market_month_list = market_month_to_train.index.tolist()
      #   storage_path = "./storage"
      exo_data = ingest_exogenous_data(dir_path)
      # print(exo_data)
    #   df_list = []
      for market_month in market_month_list:
          
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
          exo_col = get_exo_columns(prep_data)

          # condition for make decision for the number of prediction days
          print("get max pred days")
          # print(market_month_to_train.loc["AUG 24", "max_pred_day"])
          max_pred_days = market_month_to_train.loc[market_month, "max_pred_day"]
          print(max_pred_days)
          horizon = n_futures # number of prediction days default=30 days
          
          if max_pred_days < horizon:
            print("--------max_pred_days < horizon-------")
            horizon = max_pred_days
          else:
            horizon = n_futures

          
          train_df = prep_data
          # horizon = n_futures
          # print("----------------------")
          # print(horizon)
          # print("----------------------")

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
                    max_steps=epochs,
                    # early_stop_patience_steps=5,
                    loss=MAE()),
                DilatedRNN(h = horizon,
                    input_size = -1,
                    encoder_hidden_size=128,
                    hist_exog_list = exo_col, # <- Historical exogenous variables
                    scaler_type = 'robust',
                    # dropout_prob_theta=0.5,
                    max_steps=epochs,
                    # early_stop_patience_steps=5,
                    loss=MAE())
                ]
          nf = NeuralForecast(models=models, freq='D')
          nf.fit(df=train_df)
          nf.save(path=f'./checkpoints/{market_month}/',
          model_index=None, 
          overwrite=True,
          save_dataset=True)