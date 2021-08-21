
import json
import csv
import numpy as np
from apps.nlp_core.DeepMoji.deepmoji.sentence_tokenizer import SentenceTokenizer
from apps.nlp_core.DeepMoji.deepmoji.model_def import deepmoji_emojis
from apps.nlp_core.DeepMoji.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from .index import NLP_Model
from emoji import emojize

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30
batch_size = 32

class DeepMojiTokenizer(NLP_Model):
    def __init__(self, name="deepmoji",initialized=False):
        super().__init__(initialized);
        self.maxlen = 30;
    def predict(self):
        super().predict()
        self.results = self.tokenized
    def initialize(self):
        super().initialize();
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
        print(self.results)
        phrases_json = {}
        label_json = {}
        response_dict={}
        for count,phrase in enumerate(self.phrases):
            phrases_json[count]=phrase
            label_json[count]=[ str(label) for label in self.results[count]]
        response_dict['phrases'] = phrases_json
        response_dict['labels'] = label_json
        json_response = json.dumps(response_dict)
        return json_response

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
            print(ind_top)
            confidence_json[count] = int(sum(t_prob[ind_top])*100)
            # label_json[count] = ind_top.tolist()
            scores_json[count] = [int(100*t_prob[ind]) for ind in ind_top]
            label_json[count] = [ emojize(emoji_dict[str(ind)], use_aliases=True) for ind in ind_top]
            print(label_json)
        # json_response = to_json(results)
        response_dict = {}
        response_dict['phrases'] = phrases_json
        response_dict['confidence']=confidence_json
        response_dict['labels'] = label_json
        response_dict['scores'] = scores_json
        json_response = json.dumps(response_dict)
        return json_response;

emoji_dict = {
    "0":":joy:",
    "1":":unamused:",
    "2":":weary:",
    "3":":sob:",
    "4":":heart_eyes:",
    "5":":pensive:",
    "6":":ok_hand:",
    "7":":blush:",
    "8":":heart:",
    "9":":smirk:",
    "10":":grin:",
    "11":":notes:",
    "12":":flushed:",
    "13":":100:",
    "14":":sleeping:",
    "15":":relieved:",
    "16":":relaxed:",
    "17":":raised_hands:",
    "18":":two_hearts:",
    "19":":expressionless:",
    "20":":sweat_smile:",
    "21":":pray:",
    "22":":confused:",
    "23":":kissing_heart:",
    "24":":heart:",
    "25":":neutral_face:",
    "26":":information_desk_person:",
    "27":":disappointed:",
    "28":":see_no_evil:",
    "29":":weary:",
    "30":":v:",
    "31":":sunglasses:",
    "32":":rage:",
    "33":":thumbsup:",
    "34":":cry:",
    "35":":sleepy:",
    "36":":yum:",
    "37":":triumph:",
    "38":":hand:",
    "39":":mask:",
    "40":":clap:",
    "41":":eyes:",
    "42":":gun:",
    "43":":persevere:",
    "44":":smiling_imp:",
    "45":":sweat:",
    "46":":broken_heart:",
    "47":":green_heart:",
    "48":":musical_note:",
    "49":":speak_no_evil:",
    "50":":wink:",
    "51":":skull:",
    "52":":confounded:",
    "53":":smile:",
    "54":":stuck_out_tongue_winking_eye:",
    "55":":angry:",
    "56":":no_good:",
    "57":":muscle:",
    "58":":punch:",
    "59":":purple_heart:",
    "60":":sparkling_heart:",
    "61":":blue_heart:",
    "62":":grimacing:",
    "63":":sparkles:"
}
