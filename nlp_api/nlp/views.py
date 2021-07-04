from django.shortcuts import render
from django.http import HttpResponse
# from https://www.tensorflow.org/guide/keras/save_and_serialize
import numpy as np
from tensorflow import keras
import os
import pandas as pd;
print(os.getcwd())
model = keras.models.load_model('./nlp/model')
# print(model)

from sklearn.feature_extraction.text import CountVectorizer;


from . import preprocessing
from . import vectorization
from . import twitterAPIKeys as t


words_df = pd.read_csv('./nlp/All_Beauty10000.csv')
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

def getSentimentOnPhrases(phrases):
    prepped = [ preprocessing.preprocessForSentimentAnalsis(phrase, preprocessing.stopwords, preprocessing.lemmatizer) for phrase in phrases]
    actual_words= [getActuallyUsedWords(doc) for doc in prepped]
    prepped= [" ".join(doc) for doc in prepped]
    sparce_inputs = v.transform(prepped).toarray()
    print(sparce_inputs)
    output =  model.predict(sparce_inputs)
    output = list(np.squeeze(output))
    # print(output)
    return  pd.DataFrame(data= {"Phrases" : phrases,"Trained Words":actual_words,"Output" : output})

# Create your views here.
def evaluateTopic(request,twitterQuery):
    print(twitterQuery)
    print(request.headers)
    print(type(request))
    score = runOnSample(twitterQuery)
    response = HttpResponse(f'{twitterQuery} | {score[0][0]}')
    response['query']=twitterQuery
    response['score']=score[0][0]
    return response;

def evaluateTopics(request):
    print(request.GET['strings'])
    phrases = request.GET.get('strings')
    print(phrases.split(",,"))
    results = getSentimentOnPhrases(phrases.split(",,"))
    print(results)
    return render(request, 'nlp/runSample.html', {'sample':'bonk','score':1})
