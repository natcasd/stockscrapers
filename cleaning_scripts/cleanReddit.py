import pandas as pd
import numpy as np
import sqlite3
from collections import Counter
import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#from nltk.tokenize import word_tokenize

import re
import matplotlib.pyplot as plt
#from wordcloud import WordCloud, STOPWORDS


def wsb_words(date='today'):
    df = pd.read_csv('../raw_databases/reddit_wsb.csv')
    df = df.drop(columns=['created', 'id', 'url', 'comms_num'])
    df['body'] = df['body'].fillna("")
    df['text'] = df['title'] + ' ' + df['body']
    df = df.drop(columns=['body', 'title'])
    df['timestamp'] = df['timestamp'].apply(lambda x: x[0:10])

    stock = ["GME","AAL","AAPL","ABNB","ACST","AIKI","AMD","AMRN","AMRS","APHA","ASRT","ATNX","ATOS","AVGR","AZN","BIDU","BILI","BIOL","BNGO","BYND","CAN",
          "CFMS","CHFS","CIDM","CLOV","CRBP","CTRM","CTXR","DFFN","DGLY","DKNG","EBON","ECOR","FB","FCEL","FGEN","FRSX","FUTU","GEVO","HEPA","HIMX",
          "IDEX","INO","INPX","INSG","INTC","ITRM","JCS","JD","KMPH","KOPN","KXIN","LI","LKCO","MARA","MICT","MIK","MNKD","MRNA","MSFT","MU","MVIS","NAKD","NBRV","NEPT","NKLA","NNDM","NOVN","NXTD","OCGN","OGI","ONTX",
          "PDD","PERI","PLUG","POWW","PYPL","RDHL","RIOT","ROKU","SHIP","SIRI","SLGG","SNDL","SRNE","SSKN","TELL","TIGR","TLRY","TNXP","TRCH","TSLA","TXMD","UAL","VACQ","VISL","VTRS","VUZI","WIMI","WKHS","ZM"]

    #     stock = pd.read_csv('../input/stock-market-dataset/symbols_valid_meta.csv')
    #     stock = stock['Symbol'].tolist()
    #     stock_lower = [x.lower() for x in lista]
    #     stocks = stock + stock_lower
    #     words = ["an","the","of","and","a","to","in","is","you","that","it","he","was","for","on","are","as","with","his","they","I","my","than","first","water","been",
    #          "call","who","oil","its","now","find","long","down","day","did","get","come","made","may","part","some","her","would","make","like","him","into","time","has","look","two","more","write",
    #          "go","see","number","no","way","could","people","there","use","an","each","which","she","do","how","their","if","will","up","other","about","out","many","then","them","these","so","at","be",
    #          "this","have","from","or","one","had","by","word","but","not","what","all","were","we","when","your","can","said"]
    #     for element in stocks:
    #         if element in words:
    #             stocks.remove(element)

    #stock = (pd.read_csv('../input/amex-nyse-nasdaq-stock-histories/all_symbols.txt').iloc[:, 0]).to_list()
    stock = ['GME', 'AAL', 'AAPL', 'AMD', 'APHA', 'BILI',
                          'CLOV', 'DKNG', 'ECOR', 'FB', 'INO', 'JD', 'MSFT',
                          'MVIS', 'NAKD', 'PLUG', 'SNDL', 'TLRY', 'TSLA', 'WKHS', 'ZM']


    stock_lower = [x.lower() for x in stock]
    words = ["an", "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are",
             "as", "with", "his", "they", "I", "my", "than", "first", "water", "been",
             "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may",
             "part", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more",
             "write",
             "go", "see", "number", "no", "way", "could", "people", "there", "use", "an", "each", "which", "she", "do",
             "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "at",
             "be",
             "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we",
             "when", "your", "can", "said"]

    for element in stock_lower:
        if element in words:
            stock_lower.remove(element)

    for i in stock:
        df[i] = df["text"].str.contains(i, regex=False, case=False)

    group_by_timestamp = df.groupby("timestamp").agg({i: 'sum' for i in stock})
    group_by_timestamp.columns = [x.lower() for x in stock]

    generate_big_table(group_by_timestamp)

    # convert to csv and sql database
    # filtered.to_csv('reddit_with_stocks.csv')
    
    #df = pd.read_csv('reddit_with_stocks.csv')
    conn = sqlite3.connect('cleaned_reddit_twitter_stock.db')
    c = conn.cursor()

    # Insert reddit data
    group_by_timestamp.to_sql('reddit_posts_with_ticker', conn, if_exists='replace', index=True)

    # twitter and ticker data
    twitter_dataframe = pd.read_csv('./cleanedtwitterdata.csv')
    twitter_dataframe.drop_duplicates(inplace=True)
    twitter_dataframe.drop(columns=['$BBRK.B'], inplace=True)

    column_names = twitter_dataframe.columns
    twitter_dataframe.columns = [x.replace('$', '').lower() for x in column_names]

    twitter_dataframe.to_sql('twitter_posts_with_ticker', conn, if_exists='replace', index=False)

    # Insert stock data
    yahoo_1_dataframe = pd.read_csv('./yahoo_stock_1.csv')
    yahoo_1_dataframe.drop_duplicates(inplace=True)
    yahoo_1_dataframe.to_sql('yahoo_stocks_2020', conn, if_exists='replace', index=False)
    yahoo_2_dataframe = pd.read_csv('./yahoo_stock_2.csv')
    yahoo_2_dataframe.drop_duplicates(inplace=True)
    yahoo_2_dataframe.to_sql('yahoo_stock_2021', conn, if_exists='replace', index=False)


    # Close the connection to the database
    conn.close()
    




    return (df)

    #     counter = Counter(" ".join(text_clean).split()).most_common(100)

#     wordcloud = WordCloud(collocations=True).generate(' '.join(text_clean))

#      #plot the wordcloud object
#     plt.figure(figsize=(14,14))
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()
#     return(counter)


def generate_big_table(df):
    mean = list(df.mean(axis=0))
    std = list(df.std())

    list_data = df.values
    stock_list = df.columns
    timestamps = df.index

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            if num_mentions > std[col] + mean[col]:
                build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    print(build_time_stock)






wsb_words()