import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def dictionary_from_bestsellers (path, n):

    """
    
    Gives n most frequent terms in bestsellers titles
    
    """

    colnames = ['asin','title','imgUrl','productURL','stars','reviews','price',\
                'isBestSeller','boughtInLastMonth','categoryName']

    df = pd.read_csv(path, names=colnames)

    df_titles = df['title'].copy()

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df_titles)
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    return tfidf_feature_names[importance[:n]]


