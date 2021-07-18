from .models import InitialTracker
from apps.logic.model import NLP_Model

import en_core_web_sm
from apps.deepmoji.index import DeepMoji
import spacy
from spacy import displacy
import json
class SentimentTrackerAPI(NLP_Model):
    def __init__(self, name="st",initialized=False):
        super().__init__(initialized,name);
        print("Sentiment Tracker initialized")
        self.trackers = []
    def predict(self,log=False):
        super().predict()
        #Here is where we have the tracker object
        #come up with a score for each token that
        #it has found
        for tracker in self.trackers:
            tracker.getOverallSentimentForEachToken()
    def initialize(self, log = False):
        super().initialize()
        self.nlp = spacy.load("en_core_web_sm")
        self.d = DeepMoji()
        self.d.initialize()
        self.initialized = True;
    def tokenize(self,query, log = False):
        self.trackers = [InitialTracker(nlp = self.nlp, d = self.d) for q in query]
        print(self.trackers)
        super().tokenize()
        print(query)
        for count, tracker in enumerate(self.trackers):
            tracker.ScoreDoc(query[count])
    def to_json(self):
        return json.dumps([tracker.to_json() for tracker in self.trackers])
