import sklearn
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;
import pandas as pd

#this file contains all the functions needed to construct meaningful data from preprocessed words
def getAllWordsFromDF(df, col):
    return [word for words in df[col] for word in words]

def ListToString(df, col):
    return [" ".join(doc) for doc in df[col]];

def vectorize(vectorizer, vocabulary, list):
    v =vectorizer(stop_words='english')
    v.fit(vocabulary)
    sparse_vector = v.transform(list)
    return v, sparse_vector

def binarizeRating(rating):
    if rating >=3.5:
        return 1;
    return 0;
