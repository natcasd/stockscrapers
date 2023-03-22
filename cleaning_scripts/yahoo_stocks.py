import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
#from sqlalchemy import create_engine


import yfinance as yf

#pull history data for stocks
df = pd.DataFrame()
securities = ['msft', 'aapl', 'tsla', 'amzn', 'meta', 'brk-b', 'goog', 'jnj', 'jpm', 'v', 'pg',
             'ma', 'intc', 'unh', 'bac', 't', 'hd', 'xom', 'dis', 'vz', 'ko', 'mrk', 'cmcsa',
             'cvx', 'pep', 'pfe', '^gspc']

for security in securities:
   df[security] = ((yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Close\
       - yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Open)\
        /yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Close) * 100

df.to_csv('intermediate1.csv', index = True)
df = pd.read_csv('intermediate1.csv')

df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
df.set_index('Date', inplace = True)   

df2 = pd.DataFrame()
securities2 = ['gme', 'aal', 'aapl', 'amd', 'aph', 'bili', 'clov', 'dkng', 'ecor', 'meta', 'ino', 'jd', 'msft',
               'mvis', 'cenn', 'plug', 'sndl', 'tlry', 'tsla', 'wkhs', 'zm']
for security in securities2:
   df2[security] = ((yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Close\
       - yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Open)\
        /yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Close) * 100

#modify date column and set back to index
df2.to_csv('intermediate2.csv', index = True)
df2 = pd.read_csv('intermediate2.csv')

df2['Date'] = pd.to_datetime(df2['Date'].apply(lambda x: x.split()[0]))
df2.set_index('Date', inplace = True)

#save to csv 
df.to_csv('yahoo_stock_1.csv')
df2.to_csv('yahoo_stock_2.csv')

print(df)
print(df2)

