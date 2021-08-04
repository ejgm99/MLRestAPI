#this function splits a GET request into the desired phrases
#currently splits by repeated commas, which isn't the most fool
#proof method of splitting strings

from django.http import HttpResponse

def getPhrases(request):
    query = request.GET.get('strings')
    phrases = query.split(",,")
    return phrases

def runModel(model,phrases,log_performance = True,test = False):
    print("-------         IS THE MODEL INITIALIZED:    ",model.initialized)    
    if not (model.initialized):
        model.initialize()
    if (log_performance):
        model.l_tokenize(phrases)
        model.l_predict()
        json = model.to_json()
    if not log_performance:
        model.tokenize(phrases)
        model.predict()
        json = model.to_json()
    if test:
        return json
    return HttpResponse(json)
