#데이터 수집
import pandas as pd
import pandas_datareader as pdr
#kosdaq_df = pdr.get_data_yahoo('KQ11', start='2021-01-04')
#dj_df = pdr.get_data_yahoo('DJI', start='2021-01-04')
#nasdaq_df = pdr.get_data_yahoo('IXIC', start='2021-01-04')
#sp500_df = pdr.get_data_yahoo('US500', start='2021-01-04')
samsung_df = pd.rget_data_yahoo('005930.KS', start='2021-01-04')
kospi_df = pdr.get_data_yahoo('KS11', start='2021-01-04')
ce_df = pdr.get_data_yahoo('DWCCSE', start='2021-01-04')
sox_df = pdr.get_data_yahoo('SOX', start='2021-01-04')
#vkos_df = pdr.get_data_yahoo('VKOSPI', start='2021-01-04') #변동성지수 investing.com에서 가져와야 하는데 방법을 모르겠음.

samsung_df.to_csv('ss.scv')
kospi_df.to_csv('kospi.scv')
ce_df.to_csv('ce.scv')
sox_df.to_csv('sox.scv')

import pandas as pd
import pandas_datareader as pdr
import talib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import random as python_random

#데이터 불러오기
df = pd.read_csv('../../data/ss.csv',
                 index_col = 'Date',
                 parse_date = True)

kospi_df = pd.read_csv('../../data/kospi.csv',
                 index_col = 'Date',
                 parse_date = True)

sox_df = pd.read_csv('../../data/sox.csv',
                 index_col = 'Date',
                 parse_date = True)

ce_df = pd.read_csv('../../data/ce.csv',
                    index_col = 'Date',
                    parse_date = True)

#기술지표 데이터 가공
df['next_price'] = df['Adj Close'].shift(-1)
df['next_rtn'] = df['Close'] / df['Open'] -1
df['log_return'] = np.log(1 + df['Adj Close'].pct_change())
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Adj Close'], timeperiod=14)

#1.RA : Standard deviation rolling average
# Moving Average
df['MA5'] = talib.SMA(df['Close'],timeperiod=5)
df['MA10'] = talib.SMA(df['Close'],timeperiod=10)
df['RASD5'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1),timeperiod=5)
df['RASD10'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1),timeperiod=10)

#2.MACD : Moving Average Convergence/Divergence
macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd

# Momentum Indicators
#3.CCI : Commodity Channel Index
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
# Volatility Indicators

#4.ATR : Average True Range
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

#5.BOLL : Bollinger Band
upper, middle, lower = talib.BBANDS(df['Close'],timeperiod=20,nbdevup=2,nbdevdn=2,matype=0)
df['ub'] = upper
df['middle'] = middle
df['lb'] = lower

#7.MTM1
df['MTM1'] = talib.MOM(df['Close'], timeperiod=1)

#7.MTM3
df['MTM3'] = talib.MOM(df['Close'], timeperiod=3)

#8.ROC : Rate of change : ((price/prevPrice)-1)*100
df['ROC'] = talib.ROC(df['Close'], timeperiod=60)

#9.WPR : william percent range (Williams' %R)
df['WPR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

#지수시장데이터 추가
kospi_df = kospi_df.loc[:,['Close']].copy()
kospi_df.rename(columns={'Close':'KOSPI'},inplace=True)
sox_df = sox_df.loc[:,['Close']].copy()
sox_df.rename(columns={'Close':'SOX'},inplace=True)
ce_df = ce_df.loc[:,['Close']].copy()
ce_df.rename(columns={'Close':'CE'},inplace=True)

df = df.join(kospi_df,how='left')
df = df.join(sox_df,how='left')
df = df.join(ce_df,how='left')

df.head()
df.columns


# 특성목록
from sklearn.model_seclection import cross_val_score, train_test_split
from skleran.metrics import accuracy_score
feature1_list = ['Open','High','Low','Adj Close','Volume','log_return']
feature2_list = ['RASD5','RASD10','ub','lb','CCI','ATR','MACD','MA5','MA10','MTM1','MTM3','ROC','WPR']
feature3_list = ['KOSPI', 'SOX', 'CE']
feature4_list = ['next_rtn']
all_features = feature1_list + feature2_list + feature3_list + feature4_list

train,test = train_test_split
