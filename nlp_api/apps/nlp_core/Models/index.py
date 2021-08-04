import time
import json

class NLP_Model():
    def __init__(self,initialized = False,model_path = None):
        self.initialized = initialized
        self.model_path = model_path
        self.operation = False
    def LogFunctionExecutionTime(self,function, doc = None):
        if (doc == None):
            start = time.perf_counter()
            out = function()
            duration = time.perf_counter()-start
            print("Duration",duration);
        else:
            start = time.perf_counter()
            print("Performing",self.operation)
            out = function(doc)
            duration = time.perf_counter()-start
            print("Duration",duration);
        return out
    def ThrowNotYetDefined(self):
        print("This is not yet defined")
    def predict(self):
        self.operation = "predicting"
    def tokenize(self):
        self.operation = "tokenizing"
    def initialize(self):
        self.operation = "initializing"
    def to_json(self):
        self.operation = "to_json"
    def l_predict(self):
        return self.LogFunctionExecutionTime(self.predict)
    def l_tokenize(self,query):
        return self.LogFunctionExecutionTime(self.tokenize,query)
    def l_initialize(self):
        return self.LogFunctionExecutionTime(self.initialize)


class ArticleParser(NLP_Model):
    #this is a skeleton class that is meant to serve as a framework for
    #ArticleParsing and everysubject that goes into it.
    def __init__(self, name = "Article Parser"):
        super().__init__()
        self.name = name
        self.auxilary_models_initialized = False
        self.parsers_initialized = False
    def initialize(self, log = False, parser = None):
        super().initialize()
        #for this class, this will be where the
        #auxilary models will be initalized and used to define
        #the array of trackers used for document classification
        self.initialied = True
        self.auxilary_models_initialized = True
        self.parsers_initialized = True
    def tokenize(self,query, log = False):
        #query must be a list of docs (even if just one is being used)
        #so that it handles well with GET requests
        super().tokenize()
        print(self.auxilary_models_initialized,self.parsers_initialized)
        if self.auxilary_models_initialized and self.parsers_initialized:
            print("Everything initialized, parsing")
            for count, parser in enumerate(self.parsers):
                print("count")
                parser.ScoreDoc(query[count])
    def predict(self,log=False):
        super().predict()
        #Here is where we have the tracker object
        #come up with a score for each token that
        #it has found
        for parser in self.parsers:
            parser.getOverallSentimentForEachToken()
    def to_json(self):
        return json.dumps([parser.to_json() for parser in self.parsers])
