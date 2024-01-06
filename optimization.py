
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from arch import arch_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from functools import partial
from multiprocessing import Pool
from volatility import *

def expand_vol(sys_dir):
    dirct = f'{sys_dir}/data.csv'
    df = pd.read_csv(dirct)
    ticker_lst = list(set(df['ticker']))
    date_set = sorted(list(set(df['date'].values)))
    
    time_range = pd.bdate_range(start = date_set[0], end = date_set[-1])
    symbol_lst = get_symbols(sys_dir)
    for index, symbol in enumerate(symbol_lst):
        dirct = f'{sys_dir}/datas/vol_forecast/{symbol}.csv.gz'
        tmp = pd.read_csv(dirct,compression = 'gzip')
        
        tmp = tmp.reindex(time_range)
        tmp = tmp.ffill()
        tmp = tmp.fillna(0)
        
        out_dir = dirct
        tmp.to_csv(out_dir,compression = 'gzip')
    return
        

def generate_cov_mt(sys_dir, start_index, end_index):
    cov_df = pd.DataFrame({})
    symbol_lst = get_symbols(sys_dir)
    for index, symbol in enumerate(symbol_lst):
        dirct = f'{sys_dir}/datas/raw_data/{symbol}.csv.gz'
        tmp = pd.read_csv(dirct,compression = 'gzip')
        tmp = tmp[start_index,end_index]
        cov_df[symbol] = tmp['log_return'].values
    scaler = StandardScaler()
    cov_df[cov_df.columns.values.tolist()] = scaler.fit_transform(cov_df)
    cov_df.replace([np.inf,-np.inf],0,inplace = True)
    cov_mt = cov_df.cov()
    cov_values = cov_mt.values
    np.fill_diagonal(cov_values,0)
    return cov_values, scaler.var_

def gradient_cov_part(weight):
    start_index = 0
    end_index = 225
    cov_values, scale = generate_cov_mt(sys_dir,start_index, end_index)
    n = len(weight):
    gradient = [0]*n
    
    for i in range(n):
        gradient += np.multiply(weight[i]*cov_values[:,i], weight)
        
    gradient = 2*gradient
        
    return gradient, scale
    
def gradient_var_part(date,weight,scale):
    symbol_lst = get_symbols(sys_dir)
    variance = []
    for symbol in symbol_lst:
        dirct = f'{sys_dir}/datas/vol_forecast/{symbol}.csv.gz'
        tmp = pd.read_csv(dirct,compression = 'gzip')
        variance.append(tmp.loc[tmp['date'] == date]['volatility'].values**2)
    gradient = np.multiply(2*np.multiply(variance, weight**2),1/scale**2)
    return gradient
    
def gradient_return_part(date,weight,scale):
    symbol_lst = get_symbols(sys_dir)
    mean = []
    for symbol in symbol_lst:
        dirct = f'{sys_dir}/datas/return_forecast/{symbol}.csv.gz'
        tmp = pd.read_csv(dirct,compression = 'gzip')
        mean.append(tmp.loc[tmp['date'] == date]['volatility'].values)
    gradient = np.multiply(2*np.multiply(mean, weight),1/scale)
    return -gradient

def gradient_all(date,weight):
    gradient_cov, scale = gradient_cov_part(weight)
    gradient_var = gradient_var_part(date,weight,scale)
    gradient_return = gradient_return_part(date,weight,scale)
    return gradient_cov + gradient_var + gradient_return

def gradient_descent(date,weight,num_itera):
    for i in range(num_itera):
        weight += gradient_all(date,weight)
    return weight

if __name__ == '__main__':
    sys_dir = '/Users/wan/Documents/Schonfeld'
    expand_vol(sys_dir)
    symbols = get_symbols(sys_dir)
    n = len(symbols)
    num_itera = 30

    date = '20140101'
    initial_weight = ([1]*n)/n
    optimal_weight = gradient_descent(date,initial_weight,num_itera)
    
    
