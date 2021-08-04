import spacy
from .index import NLP_Model
import json

class NER_Spacy(NLP_Model):
    def initialize(self):
        super().initialize()
        self.nlp = spacy.load("en_core_web_sm")
    def tokenize(self,query):
        super().tokenize()
        self.tokenized = [self.nlp(doc) for doc in query]
    def predict(self):
        super().predict()
        self.results = [[{"text":token.text,"pos":token.pos_,"dep":token.dep_} for token in doc] for doc in self.tokenized]
    def to_json(self):
        super().to_json()
        print(self.results)
        return json.dumps(self.results)
