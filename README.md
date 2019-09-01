## LM-GEC
Build a English grammatical error correction system based on language model.

## Requirement
* python 3.4+
* pytorch 1.2
* pytorch-transformer
* Texat-pytorch
* numpy


## TODO
1. Select token to improve the minimum softmax probability of word from Bert iteratively
2. Fine-tuning train the GPT-2 in inverse word order
3. Frist use the Bert model to correct error
4. Deep bidirectional GPT-2 model to polish the output of Bert
5. Take the grammatical erroe correction as seq2seq probelm, train seq2seq model to do the GEC
