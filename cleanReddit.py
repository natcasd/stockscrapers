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
    df = pd.read_csv('./raw_databases/reddit_wsb.csv')
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

    #stocks = stock + stock_lower

    for i in stock:
        df[i] = df["text"].str.contains(i, regex=False, case=False)

    filtered = df.groupby("timestamp").agg({i: 'sum' for i in stock})

    reduced_stock_list = ['GME', 'AAL', 'AAPL', 'AMD', 'APHA', 'BILI',
                      'CLOV', 'DKNG', 'ECOR', 'FB', 'INO', 'JD', 'MSFT',
                      'MVIS', 'NAKD', 'PLUG', 'SNDL', 'TLRY', 'TSLA', 'WKHS', 'ZM']
    reduced_stock_df = filtered[reduced_stock_list]
    
    print(reduced_stock_df)



    # convert to csv and sql database
    # filtered.to_csv('reddit_with_stocks.csv')
    
    df = pd.read_csv('reddit_with_stocks.csv')

    conn = sqlite3.connect('/Users/hunteradrian/school/cs1951A/DEV-ENVIRONMENT/final-project-stock-scrapers/stocks.db')
    
    c = conn.cursor()
    
    # c.execute('''CREATE TABLE IF NOT EXISTS stocks
    #             (timestamp DATE, {} REAL)'''.format(' REAL, '.join(reduced_stock_list)))
    
    # columns = ""
    # for s in reduced_stock_df:
    #     columns += s + " REAL, "
    # columns = columns[:-2]  # remove the last comma and space
    # sql = f"CREATE TABLE IF NOT EXISTS stocks (timestamp DATE, {columns})"
    # c.execute(sql)
    
    c.execute('''CREATE TABLE IF NOT EXISTS stocks
        (timestamp DATE,
        GME REAL,
        AAL REAL,
        AAPL REAL,
        AMD REAL,
        APHA REAL,
        BILI REAL,
        CLOV REAL,
        DKNG REAL,
        ECOR REAL,
        FB REAL,
        INO REAL,
        JD REAL,
        MSFT REAL,
        MVIS REAL,
        NAKD REAL,
        PLUG REAL,
        SNDL REAL,
        TLRY REAL,
        TSLA REAL,
        WKHS REAL,
        ZM REAL)''')

    # c.execute('''CREATE TABLE IF NOT EXISTS stocks
    #             (timestamp DATE, GME REAL, AAL REAL, AAPL REAL, ABNB REAL, ACST REAL, AIKI REAL, AMD REAL, AMRN REAL, AMRS REAL, APHA REAL, ASRT REAL,
    #             ATNX REAL, ATOS REAL, AVGR REAL, AZN REAL, BIDU REAL, BILI REAL, BIOL REAL, BNGO REAL, BYND REAL, CAN REAL, CFMS REAL, CHFS REAL,
    #             CIDM REAL, CLOV REAL, CRBP REAL, CTRM REAL, CTXR REAL, DFFN REAL, DGLY REAL, DKNG REAL, EBON REAL, ECOR REAL, FB REAL, FCEL REAL,
    #             FGEN REAL, FRSX REAL, FUTU REAL, GEVO REAL, HEPA REAL, HIMX REAL, IDEX REAL, INO REAL, INPX REAL, INSG REAL, INTC REAL, ITRM REAL,
    #             JCS REAL, JD REAL, KMPH REAL, KOPN REAL, KXIN REAL, LI REAL, LKCO REAL, MARA REAL, MICT REAL, MIK REAL, MNKD REAL, MRNA REAL, MSFT REAL,
    #             MU REAL, MVIS REAL, NAKD REAL, NBRV REAL, NEPT REAL, NKLA REAL, NNDM REAL, NOVN REAL, NXTD REAL, OCGN REAL, OGI REAL, ONTX REAL,
    #             PDD REAL, PERI REAL, PLUG REAL, POWW REAL, PYPL REAL, RDHL REAL, RIOT REAL, ROKU REAL, SHIP REAL, SIRI REAL, SLGG REAL, SNDL REAL,
    #             SRNE REAL, SSKN REAL, TELL REAL, TIGR REAL, TLRY REAL, TNXP REAL, TRCH REAL, TSLA REAL, TXMD REAL, UAL REAL, VACQ REAL, VISL REAL,
    #             VTRS REAL, VUZI REAL, WIMI REAL, WKHS REAL, ZM REAL, )''')

    # Insert the data from the pandas dataframe to the SQL database
    reduced_stock_df.to_sql('stocks', conn, if_exists='replace', index=False)

    # Close the connection to the database
    conn.close()
    
    
    # conn = sqlite3.connect('test_database_for_reddit')
    # c = conn.cursor()
    # c.execute('CREATE TABLE IF NOT EXISTS redditPosts (id INT, score INT, timestamp TEXT, post TEXT, terms TEXT)')
    # conn.commit()
    # df.to_sql('redditPosts', conn, if_exists='replace', index=False)

    #with open('redditSQL.sql', 'w') as file:
     #   file.write(filedata)



    return (df)

    #     counter = Counter(" ".join(text_clean).split()).most_common(100)

#     wordcloud = WordCloud(collocations=True).generate(' '.join(text_clean))

#      #plot the wordcloud object
#     plt.figure(figsize=(14,14))
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()
#     return(counter)

wsb_words()