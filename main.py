
# https://www.cl.cam.ac.uk/research/nl/bea2019st/
# Input	Travel by bus is exspensive , bored and annoying .
# Output	Travelling by bus is expensive , boring and annoying .

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from pattern.en import lexeme
from nltk.corpus import wordnet
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')

from mxnet import gluon
from mxnet import nd
import gluonnlp


glove_6b50d = gluonnlp.embedding.create('glove', source='glove.6B.50d')
vocab = gluonnlp.Vocab(gluonnlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)

import re
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))


def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])

print(get_knn(vocab, 20, 'description'))

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

# Tokenize input
#text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#text = "travelling by bus is exspensive , bored and annoying ."
text = 'I am writing in order to express my disappointment about your musical show " Over the Rainbow " .'

tokenized_text = ['[CLS]']
doc = nlp(text)
tokenized_text.extend([token.text for token in doc])


masked_index = 10

synonyms_ = [tokenized_text[masked_index]]

for syn in wordnet.synsets(tokenized_text[masked_index]):
    for l in syn.lemmas():
        synonyms_.append(l.name())

synonyms_.extend(lexeme(tokenized_text[masked_index]))
#synonyms_.append('with')

synonyms = np.array([elm for elm in set(synonyms_)])
print(synonyms)

# Mask a token that we will try to predict back with `BertForMaskedLM`

tokenized_text[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
model.eval()

# Predict all tokens
with torch.no_grad():
    #outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    outputs = model(tokens_tensor)
    predictions = outputs[0]

token_idxs = [tokenizer.convert_tokens_to_ids([word])[0] for word in synonyms]
preds = np.array([predictions[0, masked_index, idx] for idx in token_idxs])
sort_top = preds.argsort()
predicted_index = token_idxs[sort_top[-1]]
predicted_token = synonyms[sort_top[-1]]

print('Predicted token is: ', predicted_token)

debug = 1