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


def main():
    group_by_timestamp = pd.read_csv("../cleaning_scripts/group_by_timestamp.csv", index_col=0)
    group_by_timestamp.drop_duplicates(inplace=True)
    group_by_timestamp = group_by_timestamp.astype(float)
    twitter_dataframe = pd.read_csv("../cleaning_scripts/twitter_dataframe.csv", index_col=0)
    twitter_dataframe.iloc[:,1:] = twitter_dataframe.iloc[:,1:].astype(float)
    twitter_dataframe.drop_duplicates(inplace=True)
    yahoo_1_dataframe = pd.read_csv('../cleaning_scripts/yahoo_stock_1.csv')
    yahoo_1_dataframe.drop_duplicates(inplace=True)
    yahoo_2_dataframe = pd.read_csv('../cleaning_scripts/yahoo_stock_2.csv')
    yahoo_2_dataframe.drop_duplicates(inplace=True)
    yahoo_3_dataframe = pd.read_csv('../cleaning_scripts/stock1volume.csv')
    yahoo_3_dataframe.drop_duplicates(inplace=True)
    yahoo_4_dataframe = pd.read_csv('../cleaning_scripts/stock2volume.csv')
    yahoo_4_dataframe.drop_duplicates(inplace=True)

    redditnormalized = normalize(group_by_timestamp,False)
    twitternormalized = normalize(twitter_dataframe,True)
    redditlong = reddit_lengthen(redditnormalized)
    twitterlong = twitter_lengthen(twitternormalized)
    redditmvol = reddit_merge_volatility(redditlong, yahoo_2_dataframe, yahoo_4_dataframe)
    twtrmvol = twitter_merge_volatility(twitterlong, yahoo_1_dataframe, yahoo_3_dataframe)
    
    lin_reg(twtrmvol)
    kmeans(twtrmvol)
    bins(twtrmvol)


def lin_reg(twtrmvol):
    train, test = train_test_split(twtrmvol)
    trainX = sm.add_constant(train['num_mentions'])
    testX = sm.add_constant(test['num_mentions'])
    model = sm.OLS(train['dayplus1vol'],trainX).fit()
    print(model.summary())
    trainpredicted = model.predict(trainX)
    testpredicted = model.predict(testX)

    mse_train = sm.tools.eval_measures.mse(train['dayplus1vol'],trainpredicted)
    mse_test = sm.tools.eval_measures.mse(test['dayplus1vol'],testpredicted)
    rsquared_val = r2_score(test['dayplus1vol'],testpredicted)
    print(f'mse_train: {round(mse_train,2)}, mse_test: {round(mse_test,2)}, rsquared_test: {round(rsquared_val,2)}')

    ax = twtrmvol.plot(x = 'num_mentions', y = 'dayplus1vol', kind='scatter', s=10)
    ax.set_xlabel('Number of mentions (normalized)')
    ax.set_ylabel('Stock Volatility 1 day later (%)')
    #ax.set_title("Analysis of Social Media Mentions vs. Stock Volatility", fontweight='bold')
    abline_plot(model_results=model, ax=ax, color='black', linewidth=1.3)
    plt.show()


def kmeans(twtrmvol):
    K=5
    features3d = twtrmvol[['num_mentions', 'dayplus1vol', 'dailyvolume']].to_numpy()
    kmeans = KMeans(n_clusters=K).fit(features3d) #X is 2d array (num_samples, num_features)
    clusters1, centroid_indices1 = kmeans.cluster_centers_, kmeans.labels_
    plot_features_clusters(data=features3d,centroids=clusters1,centroid_indices=centroid_indices1, threeD=True)
    features2d = twtrmvol[['num_mentions', 'dayplus1vol']].to_numpy()
    kmeans = KMeans(n_clusters=K).fit(features2d) #X is 2d array (num_samples, num_features)
    clusters2, centroid_indices2 = kmeans.cluster_centers_, kmeans.labels_
    plot_features_clusters(data=features2d,centroids=clusters2,centroid_indices=centroid_indices2, threeD=False)


def bins(df):
    #plots twitter data in 3d scatter plot #mentions, volatility, and daily volume , 
    #colors points based on bin of market cap that it falls into
    max = df['Market Cap'].max()
    bins = [0, 82000000000, 171000000000, 378000000000, max]
    df['bin'] = pd.cut(df['Market Cap'], bins=5, labels=[0,1,2,3,4])
    #print(df['bin'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:red']
    for label in [0,1,2,3,4]:
        just_label = df.loc[df['bin'] == label]
        ax.scatter(just_label['dayplus1vol'], just_label['num_mentions'], just_label['dailyvolume'], c=colors[label])

    #0 - 520Billion
    # 2.6 Trill
    # 0 - 100 mill,  100mill - 500 mill, 500 mill - 1 bil

    plt.legend(["0 - 520B", "520B - 1.04T", "1.04T - 1.56T", "1.56T-2.08T", "2.08T - 2.6T"])
    ax.set_ylabel('# of Mentions (Normalized)')
    ax.set_xlabel('Volatility 1 Day Later (%)')
    ax.set_zlabel('Volume (100M)')
    #ax.set_title('Market Cap Visualization')
    plt.show()
    """ 
    MAX_CLUSTERS = 10
    cmap = cm.get_cmap('tab10', MAX_CLUSTERS)
    colors = [cmap(l / 10) for l in df['bin']]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['num_mentions'], df['dayplus1vol'], df['dailyvolume'], c=colors)
    ax.set_xlabel('Number of mentions')
    ax.set_ylabel('Volatility 1 Day Later (%)')
    ax.set_zlabel('Volume')
    ax.set_title('Market Cap Visualization')
    fig.colorbar(ax.scatter(df['num_mentions'], df['dayplus1vol'], df['dailyvolume'], c=colors))
    plt.show()
    """

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
    msk = np.random.rand(len(df)) < train_pct
    return df[msk], df[~msk]

def plot_features_clusters(data, centroids=None, centroid_indices=None, threeD=True):

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

    ax.set_xlabel('# of Mentions (Normalized)')
    ax.set_ylabel('Volatility 1 Day Later (%)')
    if threeD:
        ax.set_zlabel('Volume (100M)')

    ax.set_title('KMeans Visualization')
    
    # Helps visualize clusters
    plt.gca().invert_xaxis()
    plt.show()
    

def reddit_merge_volatility(new_df, yahoo_2_dataframe, yahoo_4_dataframe):
    #merges volatility, market cap, and volume data from yahoo and external with reddit mention data
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


def twitter_merge_volatility(new_df, yahoo_1_dataframe, yahoo_3_dataframe):
    #merges volatility, market cap, and volume data from yahoo and external with twitter mention data
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
    yahoo_1_dataframe['Date'] = pd.to_datetime(yahoo_1_dataframe['Date'])
    yahoo_3_dataframe['Date'] = pd.to_datetime(yahoo_3_dataframe['Date'])
    new_df['dayplus1'] = new_df['timestamp'].apply(lambda x: x + timedelta(days=1))
    new_df['dayminus1'] = new_df['timestamp'].apply(lambda x: x - timedelta(days=1))
    new_df['dayplus1vol'] = ''
    new_df['dayminus1vol'] = ''

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
    #flattens reddit mention dataset into 1 column
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


main()