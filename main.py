
# https://www.cl.cam.ac.uk/research/nl/bea2019st/
# Input	Travel by bus is exspensive , bored and annoying .
# Output	Travelling by bus is expensive , boring and annoying .

import numpy as np
import spacy
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, \
                                    RobertaTokenizer, RobertaForMaskedLM,  \
                                    XLNetTokenizer, XLNetPreTrainedModel, \
                                    XLNetLMHeadModel, \
                                    XLMPreTrainedModel , XLMModel, XLMWithLMHeadModel
from forecast_token import forecast_token
import re
from googletrans import Translator
translator = Translator()
aa = translator.translate('程开甲，男，汉族，中共党员、九三学社社员，'
                          '1918年8月生，2018年11月去世，江苏吴江人，'
                          '原国防科工委科技委常任委员，中国科学院院士。')
#print(aa.text)

nlp = spacy.load('en_core_web_lg')

# Load pre-trained model tokenizer (vocabulary)
# bert-large-uncased-whole-word-masking,
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#model = RobertaForMaskedLM.from_pretrained('roberta-large')
#tokenizer = XLNetTokenizer.from_pretrained('xlm-mlm-en-2048')
#model = XLMPreTrainedModel.from_pretrained('xlm-mlm-en-2048')
model.eval()

# Tokenize input
#text = 'I am writing in order to express my disappointment about your musical show " Over the Rainbow " .'
#text = 'I am writing in order to express my disappointed about your musical show " Over the Rainbow " .'
text = "I saws the show 's advertisement hanging up of a wall in London where I was spending my holiday with some friends . " \
       "I convinced them to go there with me because I had heard good references about your Company and , " \
       "above all , about the main star , Danny Brook ."
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
for sent in sentences:
    sent_doc = nlp(sent)
    tokens = [token.text for token in sent_doc]
    for masked_index in np.arange(len(sent_doc))+1:
        if masked_index>1 and tokens[masked_index-1].istitle():  # deflautly think the word with first letter is upppercase is
            f_token, softmax_prob = tokens[masked_index-1], 100
        else:
            f_token, softmax_prob = forecast_token(sent, masked_index, tokenizer, model)
        print('Predicted token is:  ', f_token, '       softmax_prob:   ', softmax_prob)

debug = 1