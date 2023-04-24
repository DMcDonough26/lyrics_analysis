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

# step 1
def matchname(x):
    return '/'+x.lower().replace(' ','-')+'/'

def get_data():
    # load the csvs
    lyrics_df = pd.read_csv('lyrics.csv')
    artists_df = pd.read_csv('artists-data.csv')

    # match names
    artists_df['Artists_join'] = artists_df['Artist'].apply(matchname)

    return artists_df, lyrics_df

def keepval(x):
    if x['Artist_x']!=0:
        return x['Artist_x']
    elif x['Artist_y']!=0:
        return x['Artist_y']
    elif x['Artist']!=0:
        return x['Artist']
    else:
        return 'NA'

def genre(x):
    if ((x['rock_ind'] == 1) & (x['pop_ind']==1)):
        return '4. Rock/Pop'
    elif ((x['rock_ind'] == 1) & (x['rap_ind']==1)):
        return '5. Rock/Rap'
    elif ((x['pop_ind'] == 1) & (x['rap_ind']==1)):
        return '6. Pop/Rap'
    elif (x['rock_ind'] == 1):
        return '1. Rock'
    elif (x['pop_ind'] == 1):
        return '2. Pop'
    elif (x['rap_ind']==1):
        return '3. Rap'

def create_genres(artists_df, lyrics_df):
    # create genre tables
    rock_df = artists_df[artists_df['Genre']=='Rock'][['Artists_join','Artist']]
    rock_df['rock_ind'] = 1
    pop_df = artists_df[artists_df['Genre']=='Pop'][['Artists_join','Artist']]
    pop_df['pop_ind'] = 1
    rap_df = artists_df[artists_df['Genre']=='Hip Hop'][['Artists_join','Artist']]
    rap_df['rap_ind'] = 1

    # merge
    lyrics_df = lyrics_df.merge(rock_df[['Artists_join','rock_ind','Artist']],how='left',left_on='ALink',right_on='Artists_join')
    lyrics_df = lyrics_df.merge(pop_df[['Artists_join','pop_ind','Artist']],how='left',left_on='ALink',right_on='Artists_join')
    lyrics_df = lyrics_df.merge(rap_df[['Artists_join','rap_ind','Artist']],how='left',left_on='ALink',right_on='Artists_join')

    # create reduced table
    english_df = lyrics_df[((lyrics_df['rock_ind']==1)|(lyrics_df['pop_ind']==1)|(lyrics_df['rap_ind']==1))&\
                           (lyrics_df['Idiom']=='ENGLISH')].copy()

    # create single artist field
    english_df.fillna(0,inplace=True)
    english_df['artist_final'] = english_df.apply(keepval,axis=1)
    english_df.drop(['Artist_x','Artist_y','Artist','Artists_join_x','Artists_join_y','Artists_join'],axis=1,inplace=True)

    # apply single genre description
    english_df['genre'] = english_df.apply(genre,axis=1)
    english_df.drop(['rock_ind','pop_ind','rap_ind'],axis=1,inplace=True)
    english_df['songs'] = 1

    return english_df
