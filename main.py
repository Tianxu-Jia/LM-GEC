
# https://www.cl.cam.ac.uk/research/nl/bea2019st/
# Input	Travel by bus is exspensive , bored and annoying .
# Output	Travelling by bus is expensive , boring and annoying .


from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from forecast_token import forecast_token

from googletrans import Translator
translator = Translator()
aa = translator.translate('程开甲，男，汉族，中共党员、九三学社社员，1918年8月生，2018年11月去世，江苏吴江人，原国防科工委科技委常任委员，中国科学院院士。')
#print(aa.text)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
model.eval()

# Tokenize input
#text = 'I am writing in order to express my disappointment about your musical show " Over the Rainbow " .'
text = 'I am writing in order to express my disappointed about your musical show " Over the Rainbow " .'
masked_index = 9

f_token = forecast_token(text, masked_index, tokenizer, model)
print('Predicted token is: ', f_token)

debug = 1