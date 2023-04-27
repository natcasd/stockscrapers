import pandas as pd
import numpy as np
import sqlite3
from collections import Counter
import nltk
from datetime import datetime, timedelta
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from sklearn.metrics import r2_score
from matplotlib import cm

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
                          'MVIS', 'PLUG', 'CENN', 'SNDL', 'TLRY', 'TSLA', 'WKHS', 'ZM']


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
    lower_names = [x.lower() for x in stock]

    # we changed apha to aph and fb to meta
    lower_names = ['gme', 'aal', 'aapl', 'amd', 'aph', 'bili', 'clov', 'dkng', 'ecor', 'meta', 'ino', 'jd', 'msft', 'mvis', 'cenn', 'plug', 'sndl', 'tlry', 'tsla', 'wkhs', 'zm']
    group_by_timestamp.columns = lower_names

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
    twitter_dataframe.rename(columns={'fb': 'meta'}, inplace=True)

    twitter_dataframe.to_sql('twitter_posts_with_ticker', conn, if_exists='replace', index=False)

    # Insert stock data
    yahoo_1_dataframe = pd.read_csv('./yahoo_stock_1.csv')
    yahoo_1_dataframe.drop_duplicates(inplace=True)
    yahoo_1_dataframe.to_sql('yahoo_stocks_2020', conn, if_exists='replace', index=False)
    yahoo_2_dataframe = pd.read_csv('./yahoo_stock_2.csv')
    yahoo_2_dataframe.drop_duplicates(inplace=True)
    yahoo_2_dataframe.to_sql('yahoo_stock_2021', conn, if_exists='replace', index=False)

    yahoo_3_dataframe = pd.read_csv('./stock1volume.csv')
    yahoo_4_dataframe = pd.read_csv('./stock2volume.csv')


    # Close the connection to the database
    conn.close()


    ################################################################################
    ################################################################################
    # GENERATING ANALYSIS DATA

    # LINEAR REGRESSION
    
    redditnormalized = normalize(group_by_timestamp,False)
    twitternormalized = normalize(twitter_dataframe,True)

    redditlong = reddit_lengthen(redditnormalized)
    twitterlong = twitter_lengthen(twitternormalized)
    redditmvol = reddit_merge_volatility(redditlong, yahoo_2_dataframe, yahoo_4_dataframe)
    twtrmvol = twitter_merge_volatility(twitterlong, yahoo_1_dataframe, yahoo_3_dataframe)
    print(twtrmvol.head())
    twtrmvol['dayplus1vol'] = twtrmvol['dayplus1vol'].astype(float)
    
    train, test = train_test_split(twtrmvol)
    trainX = sm.add_constant(train['num_mentions'])
    testX = sm.add_constant(test['num_mentions'])
    model = sm.OLS(train['dayplus1vol'],trainX).fit()
    trainpredicted = model.predict(trainX)
    testpredicted = model.predict(testX)

    mse_train = sm.tools.eval_measures.mse(train['dayplus1vol'],trainpredicted)
    mse_test = sm.tools.eval_measures.mse(test['dayplus1vol'],testpredicted)
    rsquared_val = r2_score(test['dayplus1vol'],testpredicted)
    print(f'mse_train {mse_train}, mse_test {mse_test}, rsquared {rsquared_val}')

    ax = twtrmvol.plot(x = 'num_mentions', y = 'dayplus1vol', kind='scatter')
    abline_plot(model_results=model, ax=ax, color='black', linewidth=2)
    plt.show()

    K=5
    #KMEANS
    features3d = twtrmvol[['num_mentions', 'dayplus1vol', 'dailyvolume']].to_numpy()
    kmeans = KMeans(n_clusters=K).fit(features3d) #X is 2d array (num_samples, num_features)
    clusters1, centroid_indices1 = kmeans.cluster_centers_, kmeans.labels_
    plot_features_clusters(data=features3d,centroids=clusters1,centroid_indices=centroid_indices1, threeD=True)

    features2d = twtrmvol[['num_mentions', 'dayplus1vol']].to_numpy()
    kmeans = KMeans(n_clusters=K).fit(features2d) #X is 2d array (num_samples, num_features)
    clusters2, centroid_indices2 = kmeans.cluster_centers_, kmeans.labels_
    plot_features_clusters(data=features2d,centroids=clusters2,centroid_indices=centroid_indices2, threeD=False)

    bins(twtrmvol)


    # HYPOTHESIS 2
    reddit_df = reddit_generate_pairs(group_by_timestamp, yahoo_2_dataframe)
    reddit_df = reddit_merge_volatility(reddit_df, yahoo_2_dataframe, yahoo_4_dataframe)
    red_day_plus_one_col = list(reddit_df['dayplus1vol'])
    red_play_minus_one_col = list(reddit_df['dayminus1vol'])

    red_day_plus_one_col = [abs(x) for x in red_day_plus_one_col]  #should do this in mergevol function
    red_play_minus_one_col = [abs(x) for x in red_play_minus_one_col]

    twitter_timestamp_stock_pairs = twitter_generate_pairs(twitter_dataframe, yahoo_1_dataframe)
    twitter_timestamp_stock_pairs = twitter_merge_volatility(twitter_timestamp_stock_pairs, yahoo_1_dataframe, yahoo_3_dataframe)
    twit_day_plus_one_col = list(twitter_timestamp_stock_pairs['dayplus1vol'])
    twit_play_minus_one_col = list(twitter_timestamp_stock_pairs['dayminus1vol'])

    avg1 = sum(twit_day_plus_one_col)/len(twit_day_plus_one_col)
    avg2 = sum(twit_play_minus_one_col)/len(twit_play_minus_one_col)

    full_plus_one = red_day_plus_one_col + twit_day_plus_one_col
    full_minus_one = red_play_minus_one_col + twit_play_minus_one_col

    avg1 = sum(full_plus_one) / len(full_plus_one)
    avg2 = sum(full_minus_one) / len(full_minus_one)

    # THE BIG TEST
    tstats, pvalue = ttest_rel(twit_day_plus_one_col, twit_play_minus_one_col)


    """
    print("T-statistics: ", tstats)
    print("p-value: ", pvalue)
    print("p-value < 0.05", pvalue < 0.05)
    """

    ################################################################################
    # HYPOTHESIS 3
    reddit_df = reddit_generate_pairs(group_by_timestamp, yahoo_2_dataframe)
    reddit_df = reddit_merge_volatility(reddit_df, yahoo_2_dataframe, yahoo_4_dataframe)
    smaller_reddit_df = reddit_df[(reddit_df['stock'] == 'aapl') |
                                  (reddit_df['stock'] == 'meta') |
                                  (reddit_df['stock'] == 'msft')]
    plus_one_red_list = list(smaller_reddit_df['dayplus1vol'])

    twitter_df = twitter_timestamp_stock_pairs
    smaller_twitter_df = twitter_df[(twitter_df['stock'] == 'aapl') |
                                  (twitter_df['stock'] == 'meta') |
                                  (twitter_df['stock'] == 'msft')]

    list_twit = list(smaller_twitter_df['dayplus1vol'])

    tstats, pvalue = ttest_ind(plus_one_red_list, list_twit)
    print("ttest: " + str(tstats))
    print("pvalue: " + str(pvalue))

    #print(smaller_twitter_df)

    ################################################################################
    ################################################################################

    return df

def bins(df):
    #only for twitter data
    max = df['Market Cap'].max()
    bins = [0, 82000000000, 171000000000, 378000000000, max]
    df['bin'] = pd.cut(df['Market Cap'], bins=5, labels=[1,2,3,4,5])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['num_mentions'], df['dayplus1vol'], df['dailyvolume'], c=df['bin'], cmap='tab10')
    ax.set_xlabel('# of mentions')
    ax.set_ylabel('Volatility 1 Day Later')
    ax.set_zlabel('Volume')
    ax.set_title('Market Cap Visualization')
    fig.colorbar(ax.scatter(df['num_mentions'], df['dayplus1vol'], df['dailyvolume'], c=df['bin'], cmap='tab10'))
    plt.show()
        

def normalize(df, isTwitter):
    df = df.copy()
    if isTwitter:
        columnlist = list(df.columns)
        for i in columnlist[1:]:
            df[i] = df[i].apply(lambda x: (x - df[i].min()) / (df[i].max() - df[i].min()))
    else:
        df.drop(columns=['plug'],inplace=True)
        columnlist = list(df.columns)
        for i in columnlist:
            df[i] = df[i].apply(lambda x: (x - df[i].min()) / (df[i].max() - df[i].min()))
    return df

def train_test_split(df, train_pct=0.8):
    """
    Input:
        - df: Pandas DataFrame
        - train_pct: optional, float
    Output:
        - train dataframe: Pandas DataFrame
        - test dataframe: Pandas DataFrame
    """
    msk = np.random.rand(len(df)) < train_pct
    return df[msk], df[~msk]

def plot_features_clusters(data, centroids=None, centroid_indices=None, threeD=True):
    """
    Visualizes the song data points and (optionally) the calculated k-means
    cluster centers.
    Points with the same color are considered to be in the same cluster.

    Optionally providing centroid locations and centroid indices will color the
    data points to match their respective cluster and plot the given centroids.
    Otherwise, only the raw data points will be plotted.

    :param data: 2D numpy array of song data
    :param centroids: 2D numpy array of centroid locations
    :param centroid_indices: 1D numpy array of centroid indices for each data point in data
    :return:
    """
    MAX_CLUSTERS = 10
    cmap = cm.get_cmap('tab10', MAX_CLUSTERS)
    def plot_songs(fig, color_map=None):
        if threeD:
            x, y, z = np.hsplit(data, 3)
            fig.scatter(x, y, z, c=color_map)
        else:
            x, y = np.hsplit(data, 2)
            fig.scatter(x, y, c=color_map)

    def plot_clusters(fig):
        if threeD:
            x, y, z = np.hsplit(centroids, 3)
            fig.scatter(x, y, z, c="black", marker="x", alpha=1, s=200)
        else:
            x, y = np.hsplit(centroids, 2)
            fig.scatter(x, y, c="black", marker="x", alpha=1, s=200)

    cluster_plot = centroids is not None and centroid_indices is not None

    if threeD:
        ax = plt.figure(num=1).add_subplot(111, projection='3d')
    else:
         ax = plt.figure(num=1).add_subplot(111, projection='rectilinear')
    colors_s = None

    if cluster_plot:
        colors_s = [cmap(l / 10) for l in centroid_indices]
        plot_clusters(ax)

    plot_songs(ax, colors_s)

    ax.set_xlabel('# of mentions')
    ax.set_ylabel('Volatility 1 Day Later')
    if threeD:
        ax.set_zlabel('Average Volume')

    ax.set_title('KMeans Visualization')
    
    # Helps visualize clusters
    plt.gca().invert_xaxis()
    plt.show()


def reddit_generate_pairs(df, yahoo_2_dataframe):
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

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])
    return new_df

def reddit_merge_volatility(new_df, yahoo_2_dataframe, yahoo_4_dataframe):
    # NATHAN CODE
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
    yahoo_2_dataframe['Date'] = pd.to_datetime(yahoo_2_dataframe['Date'])
    yahoo_4_dataframe['Date'] = pd.to_datetime(yahoo_4_dataframe['Date'])
    new_df['dayplus1'] = new_df['timestamp'].apply(lambda x: x + timedelta(days=1))
    new_df['dayminus1'] = new_df['timestamp'].apply(lambda x: x - timedelta(days=1))
    new_df['dayplus1vol'] = ''
    new_df['dayminus1vol'] = ''

    for index, rows in new_df.iterrows():
        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha'):
            volatility = yahoo_2_dataframe.loc[yahoo_2_dataframe["Date"] == rows["dayplus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayplus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayplus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha'):
            volatility = yahoo_2_dataframe.loc[yahoo_2_dataframe["Date"] == rows["dayminus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayminus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayminus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha'):
            volume = yahoo_4_dataframe.loc[yahoo_4_dataframe["Date"] == rows["timestamp"], [rows["stock"]]]
            if (len(volume.values) != 0):
                new_df.loc[index, "dailyvolume"] = volume.values[0, 0]
            else:
                new_df.loc[index, "dailyvolume"] = None

    # remove rows were volatility doesn't exist (and is NaN)
    new_df = new_df.dropna()
    new_df["dayplus1vol"] = new_df["dayplus1vol"].abs()
    new_df['dayplus1vol'] = new_df['dayplus1vol'].astype(float)
    redditstats = pd.read_csv('../stocks_2020_market_cap_and_volume.csv')
    redditstats['Stock'] = redditstats['Stock'].str.lower()
    redditstats = redditstats.rename(columns={'Stock': 'stock'})
    new_df = pd.merge(new_df, redditstats, on='stock')
    return new_df

def twitter_generate_pairs(df, yahoo_1_dataframe):
    # formatting for twitter is different than for reddit so we had to drop
    # the timestamps column
    timestamps = list(df.loc[:, "created_at"])
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    mean = list(df.mean(axis=0))
    std = list(df.std())

    list_data = df.values
    stock_list = df.columns

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            if num_mentions > std[col] + mean[col]:
                build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])

    return new_df


def twitter_merge_volatility(new_df, yahoo_1_dataframe, yahoo_3_dataframe):
    # NATHAN CODE BEGINS HERE
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
    yahoo_1_dataframe['Date'] = pd.to_datetime(yahoo_1_dataframe['Date'])
    yahoo_3_dataframe['Date'] = pd.to_datetime(yahoo_3_dataframe['Date'])
    new_df['dayplus1'] = new_df['timestamp'].apply(lambda x: x + timedelta(days=1))
    new_df['dayminus1'] = new_df['timestamp'].apply(lambda x: x - timedelta(days=1))
    new_df['dayplus1vol'] = ''
    new_df['dayminus1vol'] = ''
    print(yahoo_3_dataframe)

    for index, rows in new_df.iterrows():
        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha' and rows["stock"] != 'spy'):
            volatility = yahoo_1_dataframe.loc[yahoo_1_dataframe["Date"] == rows["dayplus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayplus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayplus1vol"] = None
        else:
            new_df.loc[index, "dayplus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha' and rows["stock"] != 'spy'):
            volatility = yahoo_1_dataframe.loc[yahoo_1_dataframe["Date"] == rows["dayminus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayminus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayminus1vol"] = None
        else:
            new_df.loc[index, "dayplus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha' and rows["stock"] != 'spy'):
            volume = yahoo_3_dataframe.loc[yahoo_3_dataframe["Date"] == rows["timestamp"], [rows["stock"]]]
            if (len(volume.values) != 0):
                new_df.loc[index, "dailyvolume"] = volume.values[0, 0]
            else:
                new_df.loc[index, "dailyvolume"] = None
        else:
            new_df.loc[index, "dailyvolume"] = None

    # remove rows were volatility doesn't exist (and is NaN)
    new_df = new_df.dropna()
    new_df["dayplus1vol"] = new_df["dayplus1vol"].abs()
    new_df['dayplus1vol'] = new_df['dayplus1vol'].astype(float)
    twitterstats = pd.read_csv('../stocks_2020_market_cap_and_volume.csv')
    twitterstats['Stock'] = twitterstats['Stock'].str.lower()
    twitterstats = twitterstats.rename(columns={'Stock': 'stock'})
    new_df = pd.merge(new_df, twitterstats, on='stock')

    return new_df

def reddit_lengthen(df):
    list_data = df.values
    stock_list = df.columns
    timestamps = df.index

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])
    return new_df

def twitter_lengthen(df):
    # formatting for twitter is different than for reddit so we had to drop
    # the timestamps column
    timestamps = list(df.loc[:, "created_at"])
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    list_data = df.values
    stock_list = df.columns

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])
    return new_df


wsb_words()