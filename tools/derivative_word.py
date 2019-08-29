

import gluonnlp
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from polyglot.text import Text, Word
import spacy
import enchant
import re
import pickle
isEnglish = enchant.Dict("en")
aa = isEnglish.check("hello")

lemmatizer = WordNetLemmatizer()

nlp = spacy.load('en_core_web_sm')

'''
doc = nlp('The only major thing to note is that lemmatize takes a biggest part of speech parameter, "pos." ')
tokens = [token.text for token in doc]
for tk in tokens:
    w1 = lemmatizer.lemmatize(tk, 'v')
    w2 = lemmatizer.lemmatize(tk, pos="a")
    w3 = lemmatizer.lemmatize(tk)
    w = {w1, w2, w3}
'''

stemmer = PorterStemmer()
glove_6b50d = gluonnlp.embedding.create('glove', source='glove.6B.50d')
vocab = gluonnlp.Vocab(gluonnlp.data.Counter(glove_6b50d.idx_to_token))

pattern = "^[A-Za-z]*[A-Za-z]$"

roots_dict = dict()
for ww in vocab.idx_to_token:
    if re.search(pattern, ww) and len(ww)>2:
        root = stemmer.stem(ww)
        if ww != root:
            print(ww)
            polarity = int(Word(ww, language="en").polarity)
            ttt = {ww: polarity}
            if root not in roots_dict:
                roots_dict[root] = []
                roots_dict[root].append(ttt)
            else:
                roots_dict[root].append(ttt)

        debug = 1
fp = open("stem-words.p", "wb")
pickle.dump(roots_dict, fp)
fp.close()

fp = open("stem-words.p", "rb")
aaaa = pickle.load(fp)
fp.close()

debug = 1