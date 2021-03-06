

import numpy as np
from pattern.en import lexeme
from nltk.corpus import wordnet
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from mxnet import nd
import gluonnlp

import pickle
import enchant
from nltk.corpus import words
#import nltk
#nltk.download()
EnglishDict = enchant.Dict("en_US")

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

fp = open("./tools/stem-words.p", "rb")
stem2words = pickle.load(fp)
fp.close()


glove_6b50d = gluonnlp.embedding.create('glove', source='glove.6B.50d')
vocab = gluonnlp.Vocab(gluonnlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)

def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(word, k=2000):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])

def get_synomyms_token(token):
    stem = stemmer.stem(token)
    synonyms_ = [token]
    if stem in stem2words:
        words = stem2words[stem]
        synonyms_.extend(words)

    w1 = lemmatizer.lemmatize(token, 'v')
    w2 = lemmatizer.lemmatize(token, pos="a")
    w3 = lemmatizer.lemmatize(token)
    w = {w1, w2, w3}
    synonyms_.extend(list(w))

    #synonyms_ = [token]

    for syn in wordnet.synsets(token):
        for l in syn.lemmas():
            synonyms_.append(l.name())

    synonyms_.extend(lexeme(token))
    synonyms = np.array([elm for elm in set(synonyms_)])

    return synonyms

def get_candidate_tokens(token):
    #spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
    spacy_stopwords = list(STOP_WORDS)
    if token in spacy_stopwords:
        return spacy_stopwords

    result_ = get_knn(token, 20)
    result = []
    for ww in result_:
        # check the string from KNN is in English dictionary
        if EnglishDict.check(ww) or ww in words.words():
            result.append(ww)

    synomyms = get_synomyms_token(token)
    result.extend(synomyms)
    #result.append('reviewing')

    return result


if __name__ == '__main__':
    aa = get_candidate_tokens('people')
    bb = get_knn('took', 100)
    print(bb)

