import time

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
            print(function)
            out = function(doc)
            duration = time.perf_counter()-start
            print("Duration",duration);
        return out
    def ThrowNotYetDefined(self):
        print("This is not yet defined")
    def predict(self):
        self.operation = "predicting"
        print("predicting")
    def tokenize(self):
        self.operation = "tokenizing"
        print("tokenizing")
    def initialize(self):
        self.operation = "initializing"
        print("initializing")
    def to_json(self):
        self.operation = "to_json"
    def l_predict(self):
        return self.LogFunctionExecutionTime(self.predict)
    def l_tokenize(self,query):
        return self.LogFunctionExecutionTime(self.tokenize,query)
    def l_initialize(self):
        return self.LogFunctionExecutionTime(self.initialize)

class SampleChild(NLP_Model):
    def __init__(self, name,initialized):
        super().__init__(name,initialized);
    def predict(self, querys, log=False):
        super().predict()
    def initialize(self,query, log = False):
        super().initialize()
    def tokenize(self,query, log = False):
        super().tokenize()
    def l_predict(self,query):
        return self.LogFunctionExecutionTime(query,self.predict)
    def l_tokenize(self,query):
        return self.LogFunctionExecutionTime(query,self.tokenize)
    def l_initialize(self,query):
        return self.LogFunctionExecutionTime(query,self.initialize)
