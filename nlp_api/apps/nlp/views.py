from django.shortcuts import render
from django.http import HttpResponse
# from https://www.tensorflow.org/guide/keras/save_and_serialize

import os
import pandas as pd;

from apps.logic.getHandler import getPhrases
from apps.logic.getHandler import runModel

print("Importing NER_Spacy: ")
from apps.nlp_core.Models.spacy import NER_Spacy

print("Importing Article parser:")
from apps.nlp_core.SentimentTracker.index import InitialArticleParser

print("importing DeepMoji: ")
from apps.nlp_core.Models.deepmoji import DeepMoji,DeepMojiTokenizer

models = {
    "deepmoji": DeepMoji("deepmoji",False),
    "ner": NER_Spacy(),
    "st": InitialArticleParser(),
    "tokenizer": DeepMojiTokenizer()
}

# Create your views here.
def index(request,model_name):
    phrases = getPhrases(request)
    print("INDEX IS RUNNING MODEL:", model_name)
    return runModel(models[model_name],phrases)
