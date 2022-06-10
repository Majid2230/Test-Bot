import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from datetime import datetime
import requests
from io import StringIO
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import yfinance as yf
import json
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)


#ملاحضه تم تقسيم البرنامج لجزئين جزء للتوقع وجزء للتنبوء
class aiTest:
    def mainfunconeandonly():
        coin = 'BTC-USD'
        df = yf.download(tickers=[coin], period='3mo',interval = "1h") #Df Stands for DataFrame ..
        df1 = df.copy()
            
        #################################################################################################################       
        train_data = df1[['Close']].iloc[: - 200, :]
        valid_data = df1[['Close']].iloc[- 200:, :]
        
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        
        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
        
        
        x_train, y_train = [], []
        for i in range(60, train_data.shape[0]):
            x_train.append(train_data[i - 60: i, 0])
            y_train.append(train_data[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = []
        for i in range(60, valid_data.shape[0]):
            x_valid.append(valid_data[i - 60: i, 0])
        x_valid = np.array(x_valid)
        x_train = x_train.reshape(x_train.shape[0], 
        x_train.shape[1], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], 
        x_valid.shape[1], 1)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, 
        input_shape=x_train.shape[1:]))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=200, batch_size=64, verbose=1)
        y_pred = model.predict(x_valid)
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = y_pred.flatten()
        
        #################################################################################################################
        
        y = df['Close'].fillna(method='ffill')
        y = y.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)
        no_inputs = 60  # المدخلات
        no_outpout = 15  # المخرجات
        X = []
        Y = []
        for i in range(no_inputs, len(y) - no_outpout + 1):
            X.append(y[i - no_inputs: i])
            Y.append(y[i: i + no_outpout])
        X = np.array(X)
        Y = np.array(Y)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(no_inputs, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(no_outpout))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=200, batch_size=64, verbose=1)
        X_ = y[- no_inputs:]
        X_ = X_.reshape(1, no_inputs, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
            
        #################################################################################################################     
        #هنا التشكيل والامور هذي
        df1.rename(columns={'Close': 'Actual'}, inplace=True)
        df1['Predicted'] = np.nan
        df1['Predicted'].iloc[- y_pred.shape[0]:] = y_pred
        df1[['Actual', 'Predicted']]
        
        
        df_past = df[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=no_outpout)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        result1 = df_future.set_index('Date')
        #results = df_past.append(df_future).set_index('Date')
            
            
        fig = plt.figure(figsize=(24, 16))
        plt.plot(df1['Actual'].iloc[- y_pred.shape[0]:], label='Actual',linewidth=3)
        plt.plot(df1.Predicted, color='green',label='Predicted', linewidth=3)
        plt.plot(result1['Forecast'], color='red',label='Forecast', linewidth=3, marker = 'o')
        plt.title(coin + ' analysis')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        sadness1 = df_future.to_markdown(tablefmt="grid")
        plt.savefig('plot1.png')
        print(sadness1)