import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine
# from dotenv import load_dotenv

class DataIngestion:
    def __init__(self, user_name, password, ip, port, DB_name, path):
        self.user_name = user_name
        self.password = password
        self.ip = ip
        self.port = port
        self.DB_name = DB_name
        self.path = path
        
        # self.query_contract_price = "SELECT data_date , market_month, price_settle  FROM stg_soybean_future_prices WHERE market_month NOT LIKE '%23%'"
        self.query_contract_price = "SELECT data_date , market_month, price_settle  FROM stg_soybean_future_prices"
        self.query_continous_price = "SELECT data_date, price FROM stg_soybean_future_continuos"
    
    def IngestContractPrice(self):
        '''
        Ingest contract price form database and save to csv file
        '''
        # os.makedirs("Data_Ingestion", exist_ok=True)
        try:   
            print("Start ingest ContractPrice data")
            chunk_size = 10000
            dfs = []

            DB_connection = f'mssql+pymssql://{self.user_name}:{self.password}@{self.ip}:{self.port}/{self.DB_name}'
            engine = create_engine(
                DB_connection
            )
            conn = engine.connect().execution_options(stream_results=True)

            for chunk_dataframe in pd.read_sql(self.query_contract_price,conn,chunksize=chunk_size):
                dfs.append(chunk_dataframe)
            full_df = pd.concat(dfs)
            os.makedirs(f"{self.path}", exist_ok=True)
            contract_data_path = Path(f"{self.path}")
            file_name = "contract_data.csv"
            full_path = os.path.join(contract_data_path,file_name)
    
            full_df.to_csv(full_path, index=False)
            print("ContractPrice Data ingestion done")

        except Exception as e:
            raise e
    
    def IngestionContinuousPrice(self):
        '''
        Ingest continuous price form database and save to csv file
        '''
        try:
            print("Start ingest ContinuousPrice data")
            chunk_size = 10000
            dfs = []

            DB_connection = f'mssql+pymssql://{self.user_name}:{self.password}@{self.ip}:{self.port}/{self.DB_name}'
            engine = create_engine(
                DB_connection
            )
            conn = engine.connect().execution_options(stream_results=True)

            for chunk_dataframe in pd.read_sql(self.query_continous_price,conn,chunksize=chunk_size):
                dfs.append(chunk_dataframe)
            full_df = pd.concat(dfs)
            os.makedirs(f"{self.path}", exist_ok=True)
            continuous_data_path = Path(f"{self.path}")
            file_name = "continuous_data.csv"
            full_path = os.path.join(continuous_data_path,file_name)
            full_df.to_csv(full_path, index=False)
            print("ContinuousPrice Data ingestion done")
        except Exception as e:
            raise e


def get_contract_price_by_market_month(dir_path,market_month:str):
    try: 
        # file_path ="./storage/contract_data.csv"
        file_name = "contract_data.csv"
        # file_path ="../storage/contract_data.csv"
        check_file = os.path.isfile(os.path.join(dir_path, file_name))
        print(check_file)
        print(os.path.join(dir_path, file_name))
        if check_file:
            temp = pd.read_csv(os.path.join(dir_path, file_name), parse_dates=["data_date"])
            market_month_list = temp["market_month"].unique()
            if market_month in market_month_list:
                df = temp[temp["market_month"]==market_month]
                return df
            else:
                raise ValueError('Invalid market month')
        else:
            return ValueError('No file exist')

    except Exception as e:
        raise e

def ingest_exogenous_data(dir_path):
    file = "futures_notcontract_soybean.xlsx"
    path = os.path.join(dir_path, file)
    futures = pd.read_excel(path)

    # % of missing value by rows were dropped
    missing_values_percentage = futures.isnull().sum(axis=1) / len(futures.columns) * 100
    rows_to_drop = missing_values_percentage[missing_values_percentage > 5].index
    cleaned_df = futures.drop(index=rows_to_drop)
    cleaned_df = cleaned_df.set_index("data_date").sort_index() # set datetime column to index and sort
    cleaned_df2 = cleaned_df.drop(columns=["usd_idx_price", "usd_idx_low", "usd_idx_hight", "selling"]) # drop usd index bc high number of missing value

    missing_values_percentage = cleaned_df2.isnull().sum() / len(cleaned_df2) * 100

    # drop column remained missing values which is more than 10% 
    # Identify columns where the percentage of missing values exceeds 10%
    columns_to_drop = missing_values_percentage[missing_values_percentage > 10].index
    cleaned_df2 = cleaned_df2.drop(columns=columns_to_drop)

    # load new usdx data
    usdx1_file = "usdx_2020.csv"
    usdx2_file = "usdx_2021.csv"
    usdx3_file = "usdx_2022.csv"
    usdx4_file = "usdx_2023.csv"
    usdx5_file = "usdx_2024.csv"
    usdx1 = pd.read_csv(os.path.join(dir_path, usdx1_file), parse_dates=["Date"])
    usdx2 = pd.read_csv(os.path.join(dir_path, usdx2_file), parse_dates=["Date"])
    usdx3 = pd.read_csv(os.path.join(dir_path, usdx3_file), parse_dates=["Date"])
    usdx4 = pd.read_csv(os.path.join(dir_path, usdx4_file), parse_dates=["Date"])
    usdx5 = pd.read_csv(os.path.join(dir_path, usdx5_file), parse_dates=["Date"])

    usdx1 = usdx1.set_index("Date").sort_index()
    usdx2 = usdx2.set_index("Date").sort_index()
    usdx3 = usdx3.set_index("Date").sort_index()
    usdx4 = usdx4.set_index("Date").sort_index()
    usdx5 = usdx5.set_index("Date").sort_index()

    usdx = pd.concat([usdx1, usdx2, usdx3, usdx4, usdx5])
    usdx = usdx.drop(columns=["Open", "High", "Low"]) # use only closed price of usdx

    # Merge with new usd index data
    merged_df = pd.concat([cleaned_df2, usdx], axis=1)
    merged_df = merged_df.dropna() # drop na after merging

    # include only global value columns
    new_df1 = merged_df.copy()
    new_df2 = merged_df.copy()
    val_tot_columns = [col for col in merged_df.columns if 'value_'  in col.lower()]
    global_columns = [col for col in merged_df.columns if 'value_total'  in col.lower()]
    global_demand_sup = new_df2[global_columns]
    new_df1 = new_df1.drop(columns=val_tot_columns)
    merged_df2 = pd.concat([new_df1, global_demand_sup], axis=1)
    

    # exclude columns which has low correlation (<0.4 and >-0.4)
    exclude_feat_list = []
    cif_columns = [col for col in merged_df.columns if 'cif'  in col.lower()]
    vol_columns = [col for col in merged_df.columns if 'volume'  in col.lower()]
    qty_col = [col for col in merged_df.columns if 'quntity'  in col.lower()]

    group_feat = [cif_columns, vol_columns, qty_col]
    for feat in group_feat:
        # print("column:  ", feat)
        fil_col = exclude_feature(merged_df2,feat)
        exclude_feat_list.extend(fil_col)

    cleaned = merged_df2.drop(columns=exclude_feat_list)

    return cleaned

def get_corr_rank(merged_df, feature_list: list, target="price_soybean-seed"):   
    feature_list.append(target)
    corr_rank = merged_df[feature_list].corr()[target]
    return corr_rank.sort_values()

def exclude_feature(merged_df, feature_list: list,target="price_soybean-seed", corr_val=0.4):
    corr_rank = get_corr_rank(merged_df,feature_list, target)
    # print(corr_rank)
    filter_corr = corr_rank[(corr_rank <= corr_val) & (corr_rank >= -corr_val)].index.tolist()
    return filter_corr

def preprocess(df):

  # neuralforecast need data which has column's name as "ds", "y" and "unique_id"
  data = df.reset_index().rename(columns={"index":"ds", "price_settle":"y"})
 
  data2 = data.copy()

  # print("column ds ",data2["ds"])

  # smoothing target columns (to predict trend) moving average
  ma30 = data2["y"].rolling(window=30, center=True, min_periods=15).mean()
  data2 = data2.drop(columns="y") # drop old column y
  data2["y"] = ma30 # create new column y (trend MA30)
  
  data2["unique_id"] = "H1"


  return data2

def get_exo_columns(df):

  exo_var = df.drop(columns=["ds","y","unique_id"])
  exo_col = exo_var.columns.tolist()

  return exo_col

def assign_rmse(model, rmse_lstm, rmse_dilatedrnn):
    if model == "LSTM":
        return rmse_lstm

    elif model == "DilatedRNN":
        return rmse_dilatedrnn
    else:
        return None