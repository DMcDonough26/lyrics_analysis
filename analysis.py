import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np
import plotly.offline as pyo
pyo.init_notebook_mode()
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sklearn.metrics.pairwise as pa
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
tqdm.pandas()
import gensim
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def uniquewords(x):
    return len(set(x))

def numchar(x):
    return sum([len(word) for word in x])

def meta_features(english_df):
    english_df['total_words'] = english_df['lyric_final'].apply(len)
    english_df['unique_words'] = english_df['lyric_final'].apply(uniquewords)
    english_df['char'] = english_df['lyric_final'].apply(numchar)

    return english_df

def genre_features(english_df):
    genre_df = english_df.groupby('genre')['total_words','unique_words','char','songs'].sum()
    genre_df['pct_unique'] = round(genre_df['unique_words'] / genre_df['total_words'],2)
    genre_df['words_per_song'] = round(genre_df['total_words'] / genre_df['songs'],0)
    genre_df['word_length'] = round(genre_df['char']/genre_df['total_words'],2)

    return genre_df

def artist_features(english_df):
    artist_plot_df = english_df.groupby(['artist_final','genre'])['total_words','unique_words','char','songs'].sum()
    artist_plot_df['pct_unique'] = round(artist_plot_df['unique_words'] / artist_plot_df['total_words'],2)
    artist_plot_df['words_per_song'] = round(artist_plot_df['total_words'] / artist_plot_df['songs'],0)
    artist_plot_df['word_length'] = round(artist_plot_df['char']/artist_plot_df['total_words'],2)
    top_250_df = artist_plot_df.sort_values(by='songs',ascending=False)[0:250]

    return top_250_df

def cluster(top_250_df):
    cluster_df = top_250_df[['pct_unique','words_per_song']].copy()

    #normalize numeric variables
    for column in cluster_df.columns:
        if cluster_df[column].dtype != type(object):
            cluster_df[column] = (cluster_df[column] - cluster_df[column].mean())/cluster_df[column].std()

    # wcss = []
    #
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters = i, init = 'k-means++',
    #                     max_iter = 400, n_init = 10, random_state = 0)
    #     kmeans.fit(cluster_df)
    #     wcss.append(kmeans.inertia_)
    #
    # #Plotting the results onto a line graph to observe 'The elbow'
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Clusters')
    # plt.ylabel('WCSS') #within cluster sum of squares
    # plt.show()

    kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                    max_iter = 400, n_init = 10, random_state = 0)
    kmeans.fit(cluster_df)

    top_250_df['k_cluster'] = kmeans.labels_

    cluster_result_df = top_250_df[['pct_unique','words_per_song','word_length','k_cluster']].groupby(['k_cluster']).mean().round(2).transpose()
    # cluster_result_df['abs_diff'] = abs(cluster_result_df[0]-cluster_result_df[1])
    cluster_result_df.index.name = 'Variable'
    cluster_result_df.columns=['Cluster 1 Mean','Cluster 2 Mean','Cluster 3 Mean']

    return top_250_df, cluster_result_df

def chgstring(x):
    return (' ').join(x)

def prep_lyrics(english_df, top_250_df):
    # turn list of words back into a string
    english_df['lyric_string'] = english_df['lyric_final'].apply(chgstring)

    # filter for top artists
    temp_list = list(top_250_df.reset_index()['artist_final'])
    english_df['top_250_ind'] = english_df['artist_final'].apply(lambda x: 1 if x in temp_list else 0)
    reduced_df = english_df[english_df['top_250_ind']==1].copy()

    # create dictionary of artist clusters
    cluster_dict = dict(zip(top_250_df.reset_index()['artist_final'],top_250_df['k_cluster']))

    # add artist cluster to data set
    reduced_df['k_cluster'] = reduced_df['artist_final'].apply(lambda x: cluster_dict[x])

    # create lyrics string
    all_word_cloud = reduced_df['lyric_string'].str.cat(sep=' ')

    return english_df, reduced_df, cluster_dict, all_word_cloud

def sentiment(x):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(x)

def get_sentiment(reduced_df):
    reduced_df['sent_dict'] = reduced_df['lyric_string'].progress_apply(sentiment)

    reduced_df['neg'] = reduced_df['sent_dict'].apply(lambda x: x['neg'])
    reduced_df['neu'] = reduced_df['sent_dict'].apply(lambda x: x['neu'])
    reduced_df['pos'] = reduced_df['sent_dict'].apply(lambda x: x['pos'])
    reduced_df['nps'] = reduced_df['pos'] - reduced_df['neg']

    return reduced_df

def plot_nps(reduced_df):
    temp_df = pd.DataFrame(reduced_df.groupby(['artist_final','k_cluster','genre'])[['neg','pos']].mean())
    temp_df['nps'] = temp_df['pos'] - temp_df['neg']
    temp_df.sort_values(by='nps',inplace=True)

    fig = px.scatter(temp_df.reset_index(), x="nps", y="k_cluster", hover_data=['artist_final','k_cluster','genre'],
                    color='genre', color_discrete_sequence= ['red','blue','purple','green','orange','yellow'])
    fig.show()

# create single string of all lyrics per artist
def lyrics(x, reduced_df):
    return reduced_df[reduced_df['artist_final']==x]['lyric_string'].str.cat(sep=' ')

def lyrics_string(top_250_df, reduced_df):
    top_250_df.reset_index(inplace=True)
    top_250_df['lyric'] = top_250_df['artist_final'].apply(lyrics, args=(reduced_df,))

    return top_250_df

def vectorize_lyrics(top_250_df, cluster):
    # vectorize lyrics
    temp_df = top_250_df[top_250_df['k_cluster']==cluster].reset_index().copy()

    tfidf_vec = TfidfVectorizer(min_df=0.1, max_df=0.95)

    features = tfidf_vec.fit_transform(temp_df['lyric'])
    X = pd.DataFrame(data=features.toarray(),
                     index=temp_df['artist_final'],
                     columns=tfidf_vec.get_feature_names())

    cosine_df = pd.DataFrame(pa.cosine_similarity(X),index=X.index.values,columns=X.index.values).round(2)

    artists = list(cosine_df.index)

    return cosine_df
