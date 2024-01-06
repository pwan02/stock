
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from arch import arch_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from functools import partial
from multiprocessing import Pool

def seperate_by_ticker(sys_dir):
    dirct = f'{sys_dir}/data.csv'
    df = pd.read_csv(dirct)
    ticker_lst = list(set(df['ticker']))
    date_set = sorted(list(set(df['date'].values)))
    
    time_range = pd.bdate_range(start = date_set[0], end = date_set[-1])
    
    for item in ticker_lst:
        tmp = df.loc[df['ticker'] == item]
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp.sort_values('date')
        tmp.index = tmp['date']
        
        tmp = tmp.reindex(time_range)
        tmp = tmp.ffill()
        tmp = tmp.fillna(0)
        
        tmp['log_return'] = np.log(tmp['last']).diff(1)*1e4
        tmp.replace([np.inf,-np.inf],0,inplace = True)
        tmp = tmp.ffill()
        tmp = tmp.fillna(0)
        
        out_dir = f'{sys_dir}/datas/raw_data/{item}.csv.gz'
        tmp.to_csv(out_dir,compression = 'gzip')
        
    return 

def get_symbols(sys_dir):
    dirct = f'{sys_dir}/data.csv'
    df = pd.read_csv(dirct)
    ticker_lst = list(set(df['ticker']))
    return ticker_lst

def time_range(sys_dir):
    dirct = f'{sys_dir}/data.csv'
    df = pd.read_csv(dirct)
    ticker_lst = list(set(df['ticker']))
    date_set = sorted(list(set(df['date'].values)))
    
    time_range = pd.bdate_range(start = date_set[0], end = date_set[-1])
    return time_range

def realized_vol(y):
    return np.sqrt(np.sum(y**2))

def sub_sample(sys_dir,symbol,sample_freq,time_range):
    dirct = f'{sys_dir}/datas/raw_data/{symbol}.csv.gz'
    df = pd.read_csv(dirct,compression = 'gzip')
    df = df.sort_values('date')
    df.index = pd.to_datetime(time_range)
    
    df = df.resample('W-MON').apply({'last':'last','log_return':realized_vol,'volume':'sum'}).reset_index()
    df.columns = ['date','last','volatility','volume']
    
    df['log_return'] = np.log(df['last']).diff(1)*1e4
    tmp.replace([np.inf,-np.inf],0,inplace = True)
    df = df.ffill()
    df = df.fillna(0)
    
    out_dir = f'{sys_dir}/datas/resample/{symbol}.csv.gz'
    df.to_csv(out_dir,compression = 'gzip')
    
    return
    
def sub_sample_all(sys_dir,ticker_lst,sample_freq):
    timerange = time_range(sys_dir)
    for symbol in ticker_lst:
        sub_sample(sys_dir,symbol,sample_freq,timerange)
    return


def pred_vol(pq,window_size,series):
    
    p = int(pq[0])
    q = int(pq[1])
    num_data = len(series)
    forecast_vol = []
    
    for i in range(num_data - window_size):
        values = series[i:i+window_size]
        if np.sum(np.abs(values)) == 0:
            forecast_vol.append(0)
            continue
        try:
            model = arch_model(values, p = p, q= q)
        except ValueError:
            forecast_vol.append(0)
            continue
        model_fit = model.fit(disp = 'off')
        predictions = model_fit.forecast(horizon = 1, reindex = False)
        tmp_vol = np.sqrt(predictions.variance.values[0][0])
        forecast_vol.append(tmp_vol)
    forecast_vol = pd.Series(forecast_vol).fillna(0).tolist()
    forecast_vol = [0]*window_size + forecast_vol
    
    return forecast_vol

def pred_mean(pq,window_size,series):
    
    p = int(pq[0])
    q = int(pq[1])
    num_data = len(series)
    forecast_return = []
    
    for i in range(num_data - window_size):
        values = series[i:i+window_size]
        if np.sum(np.abs(values)) == 0:
            forecast_return.append(0)
            continue
        try:
            model = arch_model(values, p = p, q= q)
        except ValueError:
            forecast_return.append(0)
            continue
        model_fit = model.fit(disp = 'off')
        predictions = model_fit.forecast(horizon = 1, reindex = False)
        tmp_vol = np.sqrt(predictions.mean.values[0][0])
        forecast_return.append(tmp_vol)
    forecast_return = pd.Series(forecast_return).fillna(0).tolist()
    forecast_return = [0]*window_size + forecast_return
    
    return forecast_return

def vol_forecast(symbol, sys_dir):
    pq = [1,1]
    window_size = 12
    #symbol_lst = get_symbols(sys_dir)
    
    in_dir = f'{sys_dir}/datas/resample/{symbol}.csv.gz'
    tmp = pd.read_csv(in_dir,compression = 'gzip')
    series = tmp['log_return'].values
    forecast = pred_vol(pq,window_size,series)
    tmp['pred_vol'] = forecast

    out_dir = f'{sys_dir}/datas/vol_forecast/{symbol}.csv.gz'
    tmp.to_csv(out_dir,compression = 'gzip')
    return
        
def return_forecast(symbol, sys_dir):
    pq = [1,1]
    window_size = 66
    #symbol_lst = get_symbols(sys_dir)
    
    in_dir = f'{sys_dir}/datas/raw_data/{symbol}.csv.gz'
    tmp = pd.read_csv(in_dir,compression = 'gzip')
    series = tmp['log_return'].values
    forecast = pred_mean(pq,window_size,series)
    tmp['pred_vol'] = forecast

    out_dir = f'{sys_dir}/datas/return_forecast/{symbol}.csv.gz'
    tmp.to_csv(out_dir,compression = 'gzip')
    
    return

if __name__ == '__main__':
    sys_dir = '/Users/wan/Documents/Schonfeld'
    ncores = 10
    symbols = get_symbols(sys_dir)
    func_vol = partial(vol_forecast,sys_dir = sys_dir)
    func_return = partial(return_forecast,sys_dir = sys_dir)

    sys_dir = '/Users/wan/Documents/Schonfeld'
    seperate_by_ticker(sys_dir)

    ticker_lst = get_symbols(sys_dir)
    sample_freq = 'W-MON'
    sub_sample_all(sys_dir,ticker_lst,sample_freq)

    with Pool(processes = ncores) as pool:
         pool.map(func_vol,symbols)
         pool.close()
         pool.join()

    with Pool(processes = ncores) as pool:
        pool.map(func_return,symbols)
        pool.close()
        pool.join()

