import spacy
import sys
import nltk
from nltk.tokenize import word_tokenize
import gzip
import json
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import math
count = 0
# sys.stdout.write("Download progress: %d%%   \r" % (progress) )
# sys.stdout.flush()
stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']);

def getNaNs(df):
    c = 0
    for d in df['reviewText']:
        c = c+1;
        if(type(d) == float):
            print(c)

# df =pd.read_csv('All_Beauty.csv')
# df = df[:1000]
# getNaNs(df)
# isNaN(df['reviewText'][547])
# df
# filteredDf= df[df['reviewText'].apply(isNotNaN)]
# len(filteredDf)
# math.isnan(df['reviewText'][547])

def filterOutNonString(df, column):
    words_only = df[column];
    # print(len(words_only))
    bin = [type(words)==str for words in words_only]
    not_words = [i for i, x in enumerate(bin) if x == False]
    for indx in not_words:
        df = df.drop([indx])
    return df

def isNotNaN(review_text):
    try:
        return not math.isnan(review_text);
    except(TypeError):
        return True

def getDfFromJSON(path):
    print("Getting data from JSON")
    data = []
    count = 0
    with gzip.open(path) as f:
        for l in f:
            doc = json.loads(l.strip())
            sys.stdout.write("Reviews processed: %d   \r" % (count) )
            sys.stdout.flush()
            count = count + 1
            data.append(doc)
            # data.append(json.loads(l.strip()))
    sys.stdout.write("Reviews processed: %d   " % (count) )
    sys.stdout.flush()
    # print("Generating df...")
    df = pd.DataFrame.from_dict(data);
    # print("adding Column Names to DF")
    df = df[['overall', 'reviewText','summary']];
    # print("Unfiltered df", len(df))
    df = df[df['reviewText'].apply(isNotNaN)]
    # print("Returning Filtered df", len(df))
    return df

def tokenize(review):
    # tokenization
    words =word_tokenize(review);
    #remove digits and other symbols except "@"--used to remove email
    words = [re.sub(r"[^A-Za-z@]", "", word) for word in words]
    #e websites and email address
    words = [re.sub(r'\S+com', '', word) for word in words]
    words = [re.sub(r'\S+@\S+', '', word) for word in words]
    #remove empty spaces
    words = [word for word in words if word!='']
    return words;

def lowerCaseList(words_list):
    return [word.lower() for word in words_list];

def removeStopWords(text_list, stop_list):
    return [word for word in text_list if word not in stop_list]

def lemmatizeList(words_list,lemmatizer):
    return [lemmatizer(word) for word in words_list]

def lemmatizer(word):
    return nlp(word)[0].lemma_

def preprocessForSentimentAnalsis(review,stop_words,lemmatizer):
    # print("    Input String       : ",review)
    words = tokenize(review);
    stop_words  = lowerCaseList(stop_words);
    input_words = lowerCaseList(words)
    # print("    Tokenized string   : ",input_words)
    stoplessWords = removeStopWords(input_words, stop_words);
    # print("    Stopless Words     : ",stoplessWords)
    lemmatizedWords = lemmatizeList(stoplessWords, lemmatizer)
    # print("    Lemmatized Words   : ",lemmatizedWords)
    return lemmatizedWords

def preprocessDocumentList(document_array,stop_words=stopwords,lemmatizer=lemmatizer):
    return [preprocessForSentimentAnalsys(document, stopwords, lemmatizer) for document in document_list]

def preprocess(review):
    try:
        return preprocessForSentimentAnalsis(review, stopwords, lemmatizer);
    except(TypeError):
        print("This is an error")
        print(type(review),review)
        return "  "


# print("Cleaning Negative String...");
# preprocess("This product had a very rough start!");
# print("Cleaning Positive String...");
# preprocess("I immediately knew I was going to enjoy this!");



#                                 Feature            Labels
# allWords        = ['w1', 'w2' , 'w3',  ... ,'wn'] [ 5  ]
# review1 :         [0,     0   ,  0  ,  ... ,  1 ] [ 2  ]
# review2 :         [1,     0   ,  1  ,  ... ,  0 ] [ 4  ]
#                                 :
#                                 :
# reviewn:          [0,     0   ,  0  , ... ,  1 ] [ 4  ]
# review2 :         [1,     0   ,   1,  ... ,  0 ] [  1 ]
