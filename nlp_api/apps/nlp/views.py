from django.shortcuts import render
from django.http import HttpResponse
# from https://www.tensorflow.org/guide/keras/save_and_serialize
import numpy as np
from tensorflow import keras
import os
import pandas as pd;

from apps.deepmoji import index as deepmoji
from .spacy import NER_Spacy
from apps.logic.getHandler import getPhrases
from apps.logic.getHandler import runModel
from apps.sentiment_tracker.index import InitialArticleParser

models = {
    "deepmoji": deepmoji.DeepMoji("deepmoji",False),
    "ner": NER_Spacy(),
    "st": InitialArticleParser()
}


# Create your views here.
def index(request,model_name):
    phrases = getPhrases(request)
    print("INDEX IS RUNNING MODEL:", model_name)
    return runModel(models[model_name],phrases)


# model = keras.models.load_model('./apps/nlp/model')
# # print(model)
#
# from sklearn.feature_extraction.text import CountVectorizer;
#
# from . import preprocessing
# from . import vectorization
#
# words_df = pd.read_csv('./apps/nlp/All_Beauty10000.csv')
# words_df
#
# #now we construct our CountVectorizer
#
# from sklearn.feature_extraction.text import CountVectorizer;
# from sklearn.feature_extraction.text import TfidfVectorizer;
#
# #generate a sparce array from the things
# words_df['documents'] = [" ".join(preprocessing.tokenize(doc)).split(" ") for doc in words_df['documents']]
# all_words = vectorization.getAllWordsFromDF(words_df, 'documents')
# docList= [" ".join(doc) for doc in words_df['documents']]
#
# # docList = vectorization.ListToString(words_df,'documents')
# v,sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList)

def runOnSample(input):
    prepped = preprocessing.preprocessForSentimentAnalsis(input,preprocessing.stopwords,preprocessing.lemmatizer)
    # print(prepped)
    prepped= " ".join(prepped)
    # print(prepped)
    sparce_inputs = v.transform([prepped]).toarray()
    return model.predict(sparce_inputs)


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
    prediction =  model.predict(sparce_inputs)
    prediction = list(np.squeeze(prediction))
    output = zip(phrases,actual_words)
    output = zip(output,prediction)
    return  pd.DataFrame(data= {"phrases" : phrases,"trained_words":actual_words,"output" : prediction})



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
    phrases = getPhrases(request)
    results = getSentimentOnPhrases(phrases)
    return HttpResponse(results.to_json());
