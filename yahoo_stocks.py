import pandas as pd
import matplotlib.pyplot as plt


import yfinance as yf


df = pd.DataFrame()
securities = ['msft', 'aapl', 'tsla', 'amzn', 'meta', 'brk-b', 'goog', 'jnj', 'jpm', 'v', 'pg',
             'ma', 'intc', 'unh', 'bac', 't', 'hd', 'xom', 'dis', 'vz', 'ko', 'mrk', 'cmcsa',
             'cvx', 'pep', 'pfe', '^gspc']

for security in securities:
   df[security] = ((yf.Ticker(security).history(start='2020-04-08', end = '2021-07-20').Close\
       - yf.Ticker(security).history(start='2020-04-08', end = '2021-07-20').Open)\
        /yf.Ticker(security).history(start='2020-04-08', end = '2021-07-20').Close) * 100
   
   #1/28/2021 - 08/16/2021
  
print(df)