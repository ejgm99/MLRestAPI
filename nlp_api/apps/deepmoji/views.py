from django.shortcuts import render
from django.http import HttpResponse
import json
# from __future__ import print_function, division
from . import index
from apps.logic.getHandler import getPhrases
from apps.logic.getHandler import runModel

# models = {
#     "deepmoji": index.DeepMoji("deepmoji",False)
# }

# Create your views here.
def classTests(request):
    phrases = getPhrases(request)
    return runModel(models["deepmoji"],phrases)

def evaluateTopics(request):
    phrases = getPhrases(request)
    tokenized,_,_ = index.st.tokenize_sentences(phrases)
    results = index.model.predict(tokenized)

    confidence_json = {}
    phrases_json = {}
    label_json = {}
    scores_json = {}
    for count,phrase in enumerate(phrases):
        phrases_json[count] = phrase
        t_prob = results[count]
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

    return HttpResponse(json_response);

# {
#     "phrases":{
#         "0":"this is bad",
#         "1":"this is good",
#         "2":"this is neutral"
#         },
#     "trained_words":{
#         "0":["bad"],
#         "1":["good"],
#         "2":["neutral"]
#         },
#     "output":{
#         "0":0.0499616563,
#         "1":0.9949604869,
#         "2":0.895837307
#         }
# }

def top_elements(array, k):
    ind = index.np.argpartition(array, -k)[-k:]
    return ind[index.np.argsort(array[ind])][::-1]

def to_json(results):
    return json.loads("result not implemented")
