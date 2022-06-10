from webthings import Webthings
import pandas as pd
import numpy as np
import pandas_ta as ta

class Calculationthings:
    def __init__(self,cyrptocurrency):
        self.cyrptocurrency=cyrptocurrency

    def rma(x, n, y0):
        a = (n-1) / n
        ak = a**np.arange(len(x)-1, -1, -1)
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]
    def rstrengthindex(self):
        n = 14
        pandadata = self.cyrptocurrency
        df = pandadata
        #rsi_signal = df.ta.rsi(close='close', length=n, append=False, signal_indicators=False)
        df['change'] = df['close'].diff()
        df['gain'] = df.change.mask(df.change < 0, 0.0)
        df['loss'] = -df.change.mask(df.change > 0, -0.0)
        df['avg_gain'] = Calculationthings.rma(df.gain[n+1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n+1])/n)
        df['avg_loss'] = Calculationthings.rma(df.loss[n+1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n+1])/n)
        df['rs'] = df.avg_gain / df.avg_loss
        df['rsi_14'] = 100 - (100 / (1 + df.rs))
        finalds = df['rsi_14']
        finaldf = float("{:.6f}".format(finalds.at[100]))
        return finaldf
        
    def macdcalculation(self):
        pandadata = self.cyrptocurrency
        df = pandadata[['close', 'open', 'high', 'volumefrom', 'low']]
        final = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        pd.set_option("display.max_columns", None)
        MACD_12_26_9 = df.at[100, 'MACD_12_26_9']
        MACDs_12_26_9 = df.at[100, 'MACDs_12_26_9']
        finalpd = change_percent = ((float(MACD_12_26_9)-MACDs_12_26_9)/MACDs_12_26_9)*100
        return float("{:.6f}".format(finalpd))
    
    def simplemovingavarage(self):
        pandadata = self.cyrptocurrency
        sma7 = pandadata.ta.sma(close='close', length=7, append=True)
        sma14 = pandadata.ta.sma(close='close', length=14, append=True)
        sma21 = pandadata.ta.sma(close='close', length=21, append=True)
        smat = sma7.at[100]+sma14.at[100]+sma21.at[100]
        price = pandadata['close']
        smata = float("{:.6f}".format(smat/3))
        finalsma = (((float(price[100])-smata)/smata)*100)
        return finalsma
        
    def ExponentiaMA(self):
        pandadata = self.cyrptocurrency
        EMA7 = pandadata['close'].ewm(span=7).mean()
        EMA14 = pandadata['close'].ewm(span=14).mean()
        EMA21 = pandadata['close'].ewm(span=21).mean()
        EMATotal = EMA7.at[100]+EMA14.at[100]+EMA21.at[100]
        price = pandadata['close']
        EMAta = float("{:.6f}".format(EMATotal/3))
        finalEMA = (((float(price[100])-EMAta)/EMAta)*100)
        return finalEMA
        
    def Bollingerbands(self):
        df = self.cyrptocurrency
        n = 21
        m = 2
        TP = (df['high'] + df['low'] + df['close']) / 3
        data = TP
        B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
        sigma = data.rolling(n, min_periods=n).std() 
        BU = pd.Series((B_MA + m * sigma), name='BU')
        BL = pd.Series((B_MA - m * sigma), name='BL')
        BLA = (BU+BL)/2
        BLAF = BLA.at[100]
        price = df['close']
        finalbands = ((((float(BLAF)-price[100])/price[100])*100))
        return finalbands
        
    def Finalcalculation(self,name):
        df = self.cyrptocurrency
        blbands = Calculationthings.Bollingerbands(self)
        ema = Calculationthings.ExponentiaMA(self)
        sma = Calculationthings.simplemovingavarage(self)
        macd = Calculationthings.macdcalculation(self)
        rsi = Calculationthings.rstrengthindex(self)
        tottalranking = (blbands+ema+sma+macd+rsi)/5
        df = pd.DataFrame({'Name': [name.upper()],
                   'blbands': [blbands],
                   'ema_rate': [ema],
                   'sma_rate': [sma],
                   'macd': [macd],
                   'rsi': [rsi]})
        return df