import os
import tweepy as tw

API_KEY = 'Buf0XrAJRapU6rsWiYSTfIdRF'
API_SECRET_KEY = 'sLgDzKlUL4bmEqzGX2hAhfg4k1TkqPQLCrDj6PItwuwtZPANOz'
BEARER_TOKEN ='AAAAAAAAAAAAAAAAAAAAAHA6KAEAAAAAfCOde3EotWvX6k0kirovDqltjJU%3D3VtF8YCMTNSjEHq3cM7VKPq79kJYlqegtsQfecyLqA9UaZUSxw';
ACCESS_TOKEN = '704156283809636353-dsaHO9enMjWpXaj7Lluzh1S8tlTBUOd'
ACCESS_TOKEN_SECRET = 'I2v3DEh3PId9K7cS6rLTsFSxu9v2s5juPXsShFPidXIUR'

auth = tw.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tw.API(auth, wait_on_rate_limit=True)

search_words = "#WAP"
date_since = "2020-11-16"

def getTopic(topic, nTweets):
    tweets = tw.Cursor(api.search,
              q=topic,
              lang="en",
              since=date_since).items(nTweets)
    return [tweet.text for tweet in tweets]

t = getTopic("happy",20)

from . import preprocessing as p
from . import vectorization as v

p.preprocessForSentimentAnalsis(t[4],p.stopwords,p.lemmatizer)
