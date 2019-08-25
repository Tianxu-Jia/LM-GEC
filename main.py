

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)


# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Load pre-trained model (weights)
#model = BertForMaskedLM.from_pretrained('bert-base-uncased')
#fp = open('bert.json', 'w')
#print(model, file=fp)
#fp.close()

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
with torch.no_grad():
    #outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
pred = predictions[0, masked_index, :].numpy()
sort_top = pred.argsort()
#predicted_index = torch.argmax(predictions[0, masked_index, :]).item()
predicted_index = sort_top[-5:]
predicted_token = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in predicted_index]
assert predicted_token[-1] == 'henson'
print('Predicted token is:',predicted_token)