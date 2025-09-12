import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

import torch
from transformers import BertTokenizer
import json

import os

FILE_NAME = "cmu_source.txt"
#FILE_NAME = "cmudict-0.7bsymbols.txt"

DIR_PATH = os.path.dirname(__file__)
#print(DIR_PATH)

#phonemes
SYMBOLS = ['', 'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 
           'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 
           'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 
           'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 
           'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 
           'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 
           'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

#unused for the decoding process. used only when tokenizer is not found.
def read_file(batch_size=-1):
    words = []
    phoneme_lists = []
    count = 0

    with open(os.path.join(DIR_PATH, FILE_NAME), 'r') as file:
        for line in file:
            if 0 <= batch_size == count:
                break

            parts = line.strip().split(' ')
            word = parts[0]
            phonemes = [p for p in parts[1:] if p]  #exclude empty strings

            words.append(word)
            phoneme_lists.append(phonemes)

            count += 1
    return words, phoneme_lists, count

#converts/maps the decoded output of the model to the phonmes
def decode_output(input):
    result = []

    for col in range(11):
        column = [row[col] for row in input]

        max_index = np.argmax(column)
        if max_index == 0:
            break
        result.append(SYMBOLS[max_index])

    return result

#meta data
model_id = '20231126_080644'
model_name = f'lstm_{model_id}.keras'
model = load_model(model_name)
word_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#must be matching with trainer
max_word_length = 11

while True:
    #wait
    user_input = input("Enter a word (or 'exit' to quit): ").upper()
    if user_input.lower() == 'exit':
        break

    #preprocess

    def prepare_input(user_input):
        inputs = word_tokenizer.encode_plus(user_input, add_special_tokens = True, return_tensors = 'pt')
        return inputs

    #sends to model
    def get_prediction(user_input):
        #model.eval()
        with torch.no_grad():
            inputs = prepare_input(user_input)
            output = model(**inputs)
        return output
    

    encoding = word_tokenizer(user_input, padding='max_length', truncation=True, max_length=max_word_length, return_tensors='tf')
    word_sequences_padded = encoding['input_ids']
    prediction = model.predict(word_sequences_padded)

    #print("Raw model output:", prediction)
    output = decode_output(prediction)
    print(f'{model_name}: decoded output: ', output)

