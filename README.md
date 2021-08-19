# LSTM

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

#data set 나누기
train,test = train_test_split(df, testsize=0.2) #https://blog.naver.com/bluerein_/222217870818
train_df = train
test_df = test

#정규화

def min_max_normal(tmp_df):
    eng_list = []
    sample_df = tmp_df.copy()
    for x in all_features:
        if x in feature4_list :
            continue
        series = sample_df[x].copy()
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
#         print('columns : %s , Min: %f, Max: %f' % (x, scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print
        normalized = scaler.transform(values)
        new_feature = '{}_normal'.format(x)
        eng_list.append(new_feature)
        sample_df[new_feature] = normalized
    return sample_df, eng_list

train_sample_df, eng_list = min_max_normal(train_df)
test_sample_df, eng_list = min_max_normal(test_df)

train_sample_df.head()
test_sample_df.head()

#학습 데이터와 레이블 데이터 분리
num_step = 5
num_unit = 200
def create_dateset_binary(data, feature_list, step, n):
    '''
    다음날 시종가 수익률 라벨링.
    '''
    train_xdata = np.array(data[feature_list[0:n]])

    m = np.arange(len(train_xdata) - step)
    x, y = [], []
    for i in m:
        a = train_xdata[i:(i+step)]
        x.append(a)
    x_batch = np.reshape(np.array(x), (len(m), step, n))

    train_ydata = np.array(data[[feature_list[n]]])
    for i in m + step :
        next_rtn = train_ydata[i][0]
        if next_rtn > 0 :
            label = 1
        else :
            label = 0
        y.append(label)
    y_batch = np.reshape(np.array(y), (-1,1))
    return x_batch, y_batch

    eng_list = eng_list + feature4_list
n_feature = len(eng_list)-1
x_train, y_train = create_dateset_binary(train_sample_df[eng_list], eng_list, num_step, n_feature)
x_val, y_val = create_dateset_binary(val_sample_df[eng_list], eng_list, num_step, n_feature)
x_test, y_test = create_dateset_binary(test_sample_df[eng_list], eng_list, num_step, n_feature)


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)


print(pd.DataFrame(y_train).sum())
print(pd.DataFrame(y_val).sum())
print(pd.DataFrame(y_test).sum())

x_train.shape[1]

# LSTM 모델을 생성
K.clear_session()
input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
layer_lstm_1 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01))(input_layer)
layer_lstm_1 = BatchNormalization()(layer_lstm_1)
layer_lstm_2 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01))(layer_lstm_1)
layer_lstm_2 = Dropout(0.25)(layer_lstm_2)
layer_lstm_3 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01))(layer_lstm_2)
layer_lstm_3 = BatchNormalization()(layer_lstm_3)
layer_lstm_4 = LSTM(num_unit, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01))(layer_lstm_3)
layer_lstm_4 = Dropout(0.25)(layer_lstm_4)
layer_lstm_5 = LSTM(num_unit , recurrent_regularizer = regularizers.l2(0.01))(layer_lstm_4)
layer_lstm_5 = BatchNormalization()(layer_lstm_5)
output_layer = Dense(2, activation='sigmoid')(layer_lstm_5)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())

