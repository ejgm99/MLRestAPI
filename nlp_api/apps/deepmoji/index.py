
import json
import csv
import numpy as np
from .deepmoji_tools.sentence_tokenizer import SentenceTokenizer
from .deepmoji_tools.model_def import deepmoji_emojis
from .deepmoji_tools.global_variables import PRETRAINED_PATH, VOCAB_PATH
from apps.logic.model import NLP_Model

class DeepMoji(NLP_Model):
    def __init__(self, name="deepmoji",initialized=False):
        super().__init__(initialized);
        self.maxlen = 30;
    def predict(self):
        super().predict()
        print("Making a prediction");
        self.results = self.model.predict(self.tokenized)
    def initialize(self):
        super().initialize()
        print("Initializing once...")
        self.model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        self.st = SentenceTokenizer(vocabulary, maxlen)
        self.initialized = True
    def tokenize(self,query):
        print("Tokenizing object")
        super().tokenize()
        self.phrases = query
        self.tokenized, _, _ = self.st.tokenize_sentences(query)
        return self.tokenized
    def to_json(self):
        print("Turning output into JSON")
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




OUTPUT_PATH = 'test_sentences.csv'

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit',
                  u'i am disgusted']

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


maxlen = 30
batch_size = 32

# print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
# with open(VOCAB_PATH, 'r') as f:
#     vocabulary = json.load(f)
#
# st = SentenceTokenizer(vocabulary, maxlen)
#
# tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
#
# # print('Loading model from {}.'.format(PRETRAINED_PATH))
# model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
# model.summary()

# prob = model.predict(tokenized)






# Find top emojis for each sentence. Emoji ids (0-63)
# correspond to the mapping in emoji_overview.png
# at the root of the DeepMoji repo.
# print('Writing results to {}'.format(OUTPUT_PATH))
# scores = []
# for i, t in enumerate(TEST_SENTENCES):
#     t_tokens = tokenized[i]
#     t_score = [t]
#     t_prob = prob[i]
#     ind_top = top_elements(t_prob, 5)
#     t_score.append(sum(t_prob[ind_top]))
#     t_score.extend(ind_top)
#     t_score.extend([t_prob[ind] for ind in ind_top])
#     scores.append(t_score)
#     print(t_score)
#
# with open(OUTPUT_PATH, 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
#     writer.writerow(['Text', 'Top5%',
#                      'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
#                      'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
#     for i, row in enumerate(scores):
#         try:
#             writer.writerow(row)
#         except Exception:
#             print("Exception at row {}!".format(i))
