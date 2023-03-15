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


    print(filtered)
    print(filtered.sum())


    """
    sorted_values = np.sort(df['timestamp'].unique())
    print("len dataframe: " + str(len(df)))


    lista_valori = []
    
    if date == 'today':
        df = df[df['timestamp'] == sorted_values[-1]]
    elif date == 'all':
        pass
    else:
        df = df[df['timestamp'] == date]

    jx = []
    ix = []

    for j in range(0, len(df['text'])):
        for i in ((df['text'].iloc[j]).split()):
            if i in stocks:
                jx.append(j)
                ix.append(i)

    df_termini = pd.DataFrame({"indici": jx, "valori": ix})

    lista_termini = []
    print("len dataframe: " + str(len(df)) )
    for i in range(0, len(df)):
        lista_termini.append(df_termini['valori'][df_termini['indici'] == i].tolist())

    
    #df['terms'] = lista_termini
    df['terms'] = ['0'] * len(df)
    df['terms'] = df['terms'].apply(lambda x: list(set(x)))

    df['terms'] = df['terms'].apply(lambda x: ' '.join(map(str, x)))
    df['terms'] = df['terms'].apply(lambda x: ' '.join([word for word in x.split() if word not in ('I')]))

    df['text'] = df['text'].apply(lambda x: x.lower())
    #stop = stopwords.words('english')
    #df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['text'] = df['text'].apply(lambda x: re.sub(r"http\S+", "", x))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (
    '[', ']', 'array', 'will', '######(**[click', '–', "i'm", '&#x200b;', '&nbsp;', '-', 'FOR', 'To', 'it.', '/',
    'would', 'for', 'HERE', '&#x200B;', 'Array', '*****', '-', 'So', 'If', 'since', 'In', '######(**[CLICK', 'It',
    'You', 'What', 'And', 'lot', 'Some', 'got', 'it’s', '#', 'This', '>', '*', 'Is', 'They', 'My', 'Why', 'How', 'THIS',
    'going', "I'm", 'I’m', 'get', 'IS', 'We', 'WE', '-', 'I', 'THE', 'The', 'TO', 'A', 'AND', 'NOT', '🚀🚀🚀', '🚀',
    '🚀🚀')]))
    """


    # convert to csv and sql database
    filtered.sum().to_csv('reddit_with_stocks.csv')

    conn = sqlite3.connect('test_database_for_reddit')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS redditPosts (id INT, score INT, timestamp TEXT, post TEXT, terms TEXT)')
    conn.commit()
    df.to_sql('redditPosts', conn, if_exists='replace', index=False)

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