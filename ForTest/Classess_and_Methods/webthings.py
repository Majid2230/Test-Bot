import requests
import json
import pandas as pd
from datetime import datetime

class Webthings:
    def __init__(self, cryptoname):
        self.cryptoname = cryptoname
    def get_crypto_info_json(self):
        gettinginfo = requests.get("https://min-api.cryptocompare.com/data/v2/histohour?fsym=" + self.cryptoname +"&tsym=USDT&limit=100")
        jData = gettinginfo.json()
        onlyvaliddata = jData['Data']['Data']
        return onlyvaliddata
    def get_crypto_info_pandas(self):
        try:
            gettinginfo = requests.get("https://min-api.cryptocompare.com/data/v2/histohour?fsym=" + self.cryptoname +"&tsym=USDT&limit=100")
            jData = gettinginfo.json()
            onlyvaliddata = jData['Data']['Data']
            dict1212 = json.dumps(onlyvaliddata)
            dict2121 = json.loads(dict1212)
            df2 = pd.json_normalize(dict2121) 
            df2['time'] = pd.to_datetime(df2['time'],unit='s')
            return df2
        except Exception:
            pass
            
    def getcryptocustom(self):
        try:
            pagelimit = '2000'
            gettinginfo = requests.get("https://min-api.cryptocompare.com/data/v2/histohour?fsym=" + self.cryptoname +"&tsym=USDT&limit="+ pagelimit)
            jData = gettinginfo.json()
            onlyvaliddata = jData['Data']['Data']
            dict1212 = json.dumps(onlyvaliddata)
            dict2121 = json.loads(dict1212)
            df2 = pd.json_normalize(dict2121)
            df2['time'] = pd.to_datetime(df2['time'],unit='s')
            df2.rename(columns={'time': 'Date'}, inplace=True)
            return df2
        except Exception:
            pass