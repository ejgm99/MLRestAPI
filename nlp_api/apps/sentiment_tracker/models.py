from collections import Counter
import json
import emoji
import numpy as np

def deepmoji(query):
    d.tokenize([query])
    return d.predict()

def get_score():
    l = json.loads(d.to_json())
    e = [emoji_dict[str(label)] for label in l["labels"]["0"]]
    e = [emoji.emojize(i,use_aliases=True) for i in e]
    return e

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def rawScoreToEmojis(scores):
    top_scores = top_elements(scores,5)
    print(top_scores)
    return [emoji.emojize(emoji_dict[str(score)]) for score in top_scores]

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

class ArticleParser():
    def __init__(self, name = "Not named"):
        self.name = name
        self.things = {}
        self.ent_ids = []
        self.act_ids = []
        self.numberOfThings = 0;
        self.wordThingsIDs = {}
    def GetAllThings(self):
        return self.things
    def ScoreSentence(self, sentence):
        return len(sentence)
    def ScoreDoc(self, doc):
        self.doc = doc;
    def AddNewThing(self, lemma):
#         assert(isintance(token, spacy.tokens.token.Token))
        #we know that we want to keep track of everything we're keeping track of
        self.numberOfThings+=1
        self.things[self.numberOfThings] = SentimentTracker(lemma)
        self.wordThingsIDs[lemma] = self.numberOfThings
    def IsDesiredWord(self, dep):
        desired_dependencies = ['ROOT','nsubj','dobj']
        if dep in desired_dependencies:
            return True
    def IsEntity(self, dep):
        desired_dependencies = ['nsubj','dobj']
        if dep in desired_dependencies:
            return True
    def IsAct(self,dep):
        desired_dependencies = ['ROOT']
        if dep in desired_dependencies:
            return True
    def handlePRONs(self, token):
        personal_pronouns = ["I","me"]
        if token in personal_pronouns:
            return "I"
        else:
            return "unregistered pronoun"
    def getLemma(self,token):
        if token.lemma_ =='-PRON-':
            lemma = self.handlePRONs(token.text)
            return lemma
        return token.lemma_

class InitialTracker(ArticleParser):
    def __init__(self,nlp = None, d = None):
        super().__init__()
        self.nlp = nlp
        self.d = d
    def ScoreDoc(self, doc):
        super().ScoreDoc(doc)
        #this might need to get renamed, although we are distilling
        #this object so from a storage perspective might not need to
        #get stored.
        doc = self.nlp(doc)
        #this initial tracker scores the whole sentence by just sequentially taking each sentence and
        #scoring it, without really taking into account the idea of context or anything
        for count, sentence in enumerate(doc.sents):
            sentence_score = self.ScoreSentence(sentence)
            for token in sentence:
                if(self.IsDesiredWord(token.dep_)):
                    try:
                        #see if the base form of this word has been used, an add to that already active sentiment tracker
                        sentiment_tracker_id = self.wordThingsIDs[self.getLemma(token)]
                        #get sentement tracker from the sentence
                        sentiment_tracker = self.things[sentiment_tracker_id]
                        #the sentence's score is given to the individual word
                        sentiment_tracker.newSentence(count,sentence_score)
                    except(KeyError):
                        #KeyError means there's no tracking instance of that word yet. So we make a new one
                        self.AddNewThing(self.getLemma(token))
                        self.things[self.numberOfThings].newSentence(count,sentence_score)
                    #For now, we'll just accept any key word
    def AddNewThing(self, token):
        super().AddNewThing(token)
    def getOverallSentimentForEachToken(self):
        #won't actually return anything, will just get all of the tokens to calculate their final estimates
        [sentiment_tracker.calculateOverallSentiment() for sentiment_tracker in self.getSentimentTrackers()]
    def ScoreSentence(self,sentence):
        sentence = str(sentence.text)
        self.d.tokenize([sentence])
        self.d.predict()
        return self.d.results[0]
    def getSentimentTrackers(self):
        return list(self.things.values())
    def to_json(self):
        #this function will return the indexing of the sentences (maybe?)
        #but will definitely aggregate all of the jsonized predictions
        #from each token
        prelim_json = [sentiment_tracker.to_json() for sentiment_tracker in self.getSentimentTrackers()]
        print(prelim_json)
        return json.dumps(prelim_json)

#this class is meant to track nouns or verbs
#that contribute to the overall emotion state
#of a person. Will probably only ever consist
#of a designation and an emotional weight
class SentimentTracker():
    def __init__(self, name = "Not named"):
        #there will probably be some encryption in this initialization function
        self.name =name
        self.ew = -1; #ew: emotion weight
        self.sentences = {}
        self.weight = 123;
        self.sentence_count = 0;
    def updateEmotionWeight(self, weight):
        self.ew = weight
    def newSentence(self, sentence_id,weight):
        self.sentence_count+=1
        self.sentences[sentence_id] = weight
    def calculateOverallSentiment(self):
        #calculates how the sentiment of the word evolves over the course of the document
        #this should probably be broken out so that the evolution can be explored over
        #each logged sentence. Because each sentence is ID'd with respect to the overall
        #document this shouldn't be hard. For now it is just a simple average.
        #This is one of the many examples of hand-wavy statistics that will need to be polished
        #through rigorous research and model training
        score_list = list(self.sentences.values())
        #take mean of each data point in 64-dof emotion space
        self.avg = np.mean(score_list,axis = 0)
    def to_json(self):
        jdict = {
            "name" : self.name,
            "sentences":list(self.sentences.keys()),
            "score" : rawScoreToEmojis(self.avg)
        }
        return jdict
