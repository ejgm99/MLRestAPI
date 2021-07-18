# from https://www.tensorflow.org/guide/keras/save_and_serialize
import numpy as np
from tensorflow import keras
import os
import pandas as pd;
print(os.getcwd())
model = keras.models.load_model('ml/model')
# print(model)

from sklearn.feature_extraction.text import CountVectorizer;


from . import preprocessing
from . import vectorization
from . import twitterAPIKeys as t


words_df = pd.read_csv('All_Beauty10000.csv')
words_df

#now we construct our CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;

#generate a sparce array from the things
words_df['documents'] = [" ".join(preprocessing.tokenize(doc)).split(" ") for doc in words_df['documents']]
all_words = vectorization.getAllWordsFromDF(words_df, 'documents')
docList= [" ".join(doc) for doc in words_df['documents']]

# docList = vectorization.ListToString(words_df,'documents')
v,sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList)


def runOnSample(input):
    prepped = preprocessing.preprocessForSentimentAnalsis(input,preprocessing.stopwords,preprocessing.lemmatizer)
    # print(prepped)
    prepped= " ".join(prepped)
    # print(prepped)
    sparce_inputs = v.transform([prepped]).toarray()
    return model.predict(sparce_inputs)

def getSentimentOfTopic(topic, nTweets):
    tweets = t.getTopic(topic, nTweets)
    # print(tweets)
    prepped = [ preprocessing.preprocessForSentimentAnalsis(tweet,preprocessing.stopwords,preprocessing.lemmatizer) for tweet in tweets]
    actual_words= [getActuallyUsedWords(doc) for doc in prepped]
    prepped= [" ".join(doc) for doc in prepped]
    sparce_inputs = v.transform(prepped).toarray()
    output =  model.predict(sparce_inputs)
    output = list(np.squeeze(output))
    # print(output)
    return  pd.DataFrame(data= {"Tweets" : tweets,"Trained Words":actual_words,"Output" : output})

def getActuallyUsedWords(phrase):
    out = []
    for word in phrase:
        if max(v.transform([word]).toarray()[0]) !=0:
            out = out+ [word]
    return out;

# getActuallyUsedWords(["real", "notarealword"])
#
# donald_df = getSentimentOfTopic("donald trump",20);
# d = {'col1': [1, 2], 'col2': [3, 4]}
# df = pd.DataFrame(data=d)
# df
#
# wap, wap_actual_words, wap_analysis = getSentimentOfTopic("wap", 20)
# wap_analysis
#
# happy_tweets, used, nnout = getSentimentOfTopic("happy",20)
print(os.getcwd())
