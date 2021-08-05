
import json
import csv
import numpy as np
from apps.nlp_core.DeepMoji.deepmoji.sentence_tokenizer import SentenceTokenizer
from apps.nlp_core.DeepMoji.deepmoji.model_def import deepmoji_emojis
from apps.nlp_core.DeepMoji.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from .index import NLP_Model

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30
batch_size = 32

class DeepMoji(NLP_Model):
    def __init__(self, name="deepmoji",initialized=False):
        super().__init__(initialized);
        self.maxlen = 30;
    def predict(self):
        super().predict()
        self.results = self.model.predict(self.tokenized)
    def initialize(self):
        super().initialize();
        self.model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        self.st = SentenceTokenizer(vocabulary, maxlen)
        self.initialized = True
    def tokenize(self,query):
        super().tokenize()
        self.phrases = query
        self.tokenized, _, _ = self.st.tokenize_sentences(query)
        return self.tokenized
    def to_json(self):
        super().to_json()
        confidence_json = {}
        phrases_json = {}
        label_json = {}
        scores_json = {}
        for count,phrase in enumerate(self.phrases):
            phrases_json[count] = phrase
            t_prob = self.results[count]
            ind_top = top_elements(t_prob, 5)
            confidence_json[count] = int(sum(t_prob[ind_top])*100)
            label_json[count] = ind_top.tolist()
            scores_json[count] = [int(100*t_prob[ind]) for ind in ind_top]
        # json_response = to_json(results)
        response_dict = {}
        response_dict['phrases'] = phrases_json
        response_dict['confidence']=confidence_json
        response_dict['labels'] = label_json
        response_dict['scores'] = scores_json
        json_response = json.dumps(response_dict)
        return json_response;
