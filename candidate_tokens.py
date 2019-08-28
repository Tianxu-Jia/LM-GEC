

import numpy as np
from pattern.en import lexeme
from nltk.corpus import wordnet
import spacy

from mxnet import nd
import gluonnlp

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
    synonyms_ = [token]

    for syn in wordnet.synsets(token):
        for l in syn.lemmas():
            synonyms_.append(l.name())

    synonyms_.extend(lexeme(token))
    synonyms = np.array([elm for elm in set(synonyms_)])
    return synonyms

def get_candidate_tokens(token):
    spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
    if token in spacy_stopwords:
        return spacy_stopwords

    result = get_knn(token, 20)
    synomyms = get_synomyms_token(token)
    result.extend(synomyms)
    #result.append('reviewing')

    return result


if __name__ == '__main__':
    aa = get_candidate_tokens('people')
    print(aa)

