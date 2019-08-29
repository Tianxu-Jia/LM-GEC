
from candidate_tokens import get_candidate_tokens
import torch
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')


def forecast_token(text, masked_index, tokenizer, model):
    tokenized_text = ['[CLS]']
    doc = nlp(text)
    tokenized_text.extend([token.text for token in doc])
    tokenized_text.append('SEP')

    synonyms = get_candidate_tokens(tokenized_text[masked_index])
    synonyms = list(set(synonyms))

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
    predicted_index = token_idxs[sort_top[-1]]
    predicted_token = synonyms[sort_top[-1]]
    return predicted_token

    # confirm we were able to predict 'henson'
    #predicted_index = torch.argmax(predictions[0, masked_index]).item()
    #predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    #return predicted_token
