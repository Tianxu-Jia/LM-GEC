
from candidate_tokens import get_candidate_tokens
import torch
import numpy as np
from polyglot.text import Word
import spacy
nlp = spacy.load('en_core_web_sm')


def forecast_token(text, masked_index, tokenizer, model):
    tokenized_text = ['[CLS]']
    doc = nlp(text)
    tokenized_text.extend([token.text for token in doc])
    tokenized_text.append('[SEP]')

    synonyms_ = get_candidate_tokens(tokenized_text[masked_index])
    synonyms_ = list(set(synonyms_))

    masked_token = tokenized_text[masked_index]
    token_polarity = int(Word(masked_token, language="en").polarity) #######

    synonyms = []
    for elem in synonyms_:
        if int(Word(elem, language="en").polarity) == token_polarity:
            synonyms.append(elem)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    tokenized_text[masked_index] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])


    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    token_idxs = [tokenizer.convert_tokens_to_ids([word])[0] for word in synonyms]
    preds = np.array([predictions[0, masked_index, idx] for idx in token_idxs])
    sort_top = preds.argsort()
    #predicted_index = token_idxs[sort_top[-1]]
    candiditate_tokens = [synonyms[sort_top[-1]], synonyms[sort_top[-2]]]
    if masked_token in candiditate_tokens:  # if the probability of masked token within top two, then think the masked token is correct.
        predicted_token, softmax_prob = masked_token, 100
    else:
        predicted_token, softmax_prob = synonyms[sort_top[-1]], preds[sort_top[-1]]

    # don't change the token if the predicted token is same at the original token
    # without consider the upper/lower case
    if masked_token.lower() == predicted_token.lower():
        predicted_token = masked_token
    return predicted_token, softmax_prob

