import sys  
sys.path.append('Classess_and_Methods/') 
from webthings import Webthings
from calculationthings import Calculationthings
import json
import pandas as pd



class Caller:
    def getcryptodetails():
        headerList = ['name', 'blbands_rate', 'ema_rate', 'sma_rate', 'macd_rate', 'rsi']
        with open('currencies.txt') as f:
            lines = f.readlines()
        count = 0
        for currency in lines:
            currency = currency.replace("\n", "")
            count += 1
            jsondata1 = Webthings(currency.upper())
            pandadata = jsondata1.get_crypto_info_pandas()
            try:
                sadness = Calculationthings(pandadata)
                dfsad = sadness.Finalcalculation(currency)
                with open('output/output.csv', "a") as o:
                    o.write('\n')
                dfsad.to_csv('output/output.csv', index = False, header = False, mode='a')
            except Exception:
                pass
        df_new = pd.read_csv('output/output.csv')
        GFG = pd.ExcelWriter('output/output.xlsx')
        df_new.to_excel(GFG, index=False, header = headerList)
        GFG.save()