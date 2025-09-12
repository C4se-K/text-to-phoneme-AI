import os
import time
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import AdditiveAttention
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json

from transformers import BertTokenizer

FILE_NAME = "cmu_source.txt" # data source file name
DIR_PATH = os.path.dirname(__file__)

#list of ARPABET phonemes that are used
SYMBOLS = ['', 'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 
           'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 
           'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 
           'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 
           'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 
           'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 
           'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

def calculate_statistics(list_of_lists):
    lengths = [len(sublist) for sublist in list_of_lists]
    mean = np.mean(lengths)
    median = np.median(lengths)
    std_dev = np.std(lengths)

    # Print the results
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")

    return mean, median, std_dev

def read_file(batch_size=-1):
    words = []
    phoneme_lists = []
    count = 0

    with open(os.path.join(DIR_PATH, FILE_NAME), 'r') as file:
        for line in file:
            if 0 <= batch_size == count:
                break

            parts = line.strip().split(' ')
            word = parts[0].lower()
            phonemes = [p for p in parts[1:] if p] #exclude empty strings

            words.append(word)
            phoneme_lists.append(phonemes)

            count += 1
    return words, phoneme_lists, count

"""
preprocess the data

tokenization of vocabulary & tokenization of phonemes
"""
def preprocess_data(words, phoneme_lists, max_word_length, max_phoneme_length, bert_model_name='bert-base-uncased'):
    # initialize tokenier
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # tokenize
    encoding = bert_tokenizer(words, padding='max_length', truncation=True, max_length=max_word_length, return_tensors='tf')
    word_sequences_padded = encoding['input_ids']

    #initialize tokenizer for phonemes
    phoneme_tokenizer = Tokenizer(filters='', oov_token=None)
    phoneme_tokenizer.fit_on_texts(SYMBOLS)

    # Process 
    phoneme_sequences = []
    for phoneme_list in phoneme_lists:
        filtered_phoneme_list = [p for p in phoneme_list if p in SYMBOLS]
        phoneme_sequences.append(phoneme_tokenizer.texts_to_sequences(filtered_phoneme_list))

    #pad 
    phoneme_sequences_padded = pad_sequences(phoneme_sequences, maxlen=max_phoneme_length, padding='post')

    return word_sequences_padded, phoneme_sequences_padded, bert_tokenizer, phoneme_tokenizer

"""
model architecture:

sequential models are linear models where layers are stacked on top of each other.

bidirectional -> layer that allows for the capturing of long term relationships
dropout -> prevents overfitting by randomly setting n% of values to zero
dense -> outputs the prediction, uses softmax to match the dimentions of the phonemes
compile -> configures the model for training

"""
def build_lstm_model(vocab_size, phoneme_vocab_size, max_word_length, embedding_dim=128):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_word_length))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))  #changed to return sequences
    model.add(Dense(phoneme_vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_validation_set(words, phoneme_lists):
    df = pd.DataFrame({'word': words, 'phoneme': phoneme_lists})

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)  # 80% training, 20% validation

    #x_train, y_train, x_val, y_val
    x_train = train_df['word'].tolist()
    y_train = train_df['phoneme'].tolist()
    x_val = val_df['word'].tolist()
    y_val = val_df['phoneme'].tolist()

    return x_train, y_train, x_val, y_val

def save_tokenizer(word_tokenizer, name):
    tokenizer_json = word_tokenizer.to_json()
    tokenizer_name = f'tokenizer_{name}.json'
    with open(tokenizer_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def get_model(name=""):
    if name == "":
        print('building new lstm model...')
        model = build_lstm_model(vocab_size, phoneme_vocab_size, max_word_length)
    else:
        print(f'loading model {name}...')
        model_name = f'lstm_{name}.keras'
        model = load_model(model_name)

    return model

def get_tokenizer(name=""):
    if name == "":
        print('creating new tokenizer...')
        word_tokenizer = Tokenizer(oov_token='<OOV>') 
    else:
        print(f'loading tokenizer {name}...')
        tokenizer_name = f'tokenizer_{name}.json'
        with open(os.path.join(DIR_PATH, tokenizer_name))as f:
            data = json.load(f)
            word_tokenizer = tokenizer_from_json(data)
    return word_tokenizer




start = time.time()
words, phoneme_lists, count = read_file()
print(f'there are {count} elements in the list')
print(f'statistics of the dataset: {calculate_statistics(phoneme_lists)}')
print(f'operation took {time.time()-start} seconds')

max_word_length = 20
max_phoneme_length = max_word_length
x_train, y_train, x_val, y_val = build_validation_set(words, phoneme_lists)

model_id = ''
word_tokenizer_name = 'bert-base-uncased'

word_sequences_padded, phoneme_sequences_padded, word_tokenizer, phoneme_tokenizer = preprocess_data(x_train, 
                                                                                                     y_train, 
                                                                                                     max_word_length, 
                                                                                                     max_phoneme_length, 
                                                                                                     word_tokenizer_name)

test_words_sequences_padded, test_phone_sequences_padded, test_word_tokenizer, test_phoneme_tokenizer = preprocess_data(x_val, 
                                                                                                                        y_val, 
                                                                                                                        max_word_length, 
                                                                                                                        max_phoneme_length, 
                                                                                                                        word_tokenizer_name)

vocab_size = word_tokenizer.vocab_size
phoneme_vocab_size = len(phoneme_tokenizer.word_index) + 1  # +1 for padding token

start = time.time()
model = get_model(model_id)
print(f'operation took {time.time()-start} seconds')
print('training start...')
start = time.time()

attention = AdditiveAttention(name = 'attention_layer')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

model.fit(word_sequences_padded, 
          phoneme_sequences_padded, 
          epochs=30, batch_size=32, 
          validation_data=(test_words_sequences_padded, test_phone_sequences_padded),
          callbacks = [early_stopping])

print(f'operation took {time.time()-start} seconds')
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f'lstm_{current_time}.keras'

#save the model
model.save(model_name)

