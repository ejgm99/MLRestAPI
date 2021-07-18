import logic.getHandler as g
import nlp.spacy as s

ner = s.NER_Spacy();

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit',
                  u'i am disgusted']

print(g.runModel(ner, TEST_SENTENCES,test = True))
