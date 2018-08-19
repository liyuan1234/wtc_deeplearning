#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:14:28 2018

@author: liyuan
"""
import time
start_time = time.time()

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout
from keras.models import Model
import keras
import nltk
from load_glove_embeddings import load_glove_embeddings

from wtc_utils import preprocess_data

#word2index, embedding_matrix = load_glove_embeddings('./word embeddings/glove.6B.50d.txt', embedding_dim=50)

load_data = 1
if load_data:
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,exp_intseq,lengths,cache = data
    maxlen_question,maxlen_exp,vocablen_question,vocablen_exp = lengths
    questions_vocab_idx,questions_vocab,questions,answers,answers_intseq,exp_vocab,exp_vocab_dict,exp_tokenized = cache


## convert answers to 1,2,3,4
#answers_int = [option for option,_ in answers]
#answers_int = list([convert_to_int(x) for x in answers_int])
#answers_onehot = to_categorical(answers_int)

print("--- {:.2f} seconds ---".format(time.time() - start_time))

#%% keras model

NUM_HIDDEN_UNITS = 100

input1 = Input((maxlen_exp,))
X1 = Embedding(input_dim = vocablen_exp,output_dim = NUM_HIDDEN_UNITS)(input1)
X1 = Dropout(0.5)(X1)
X1 = LSTM(NUM_HIDDEN_UNITS)(X1)
X1 = Dropout(0.5)(X1)
X1 = keras.layers.RepeatVector(maxlen_question)(X1)

input2 = Input((maxlen_question,))
X2 = Embedding(input_dim = vocablen_question,output_dim = NUM_HIDDEN_UNITS)(input2)
X2 = Dropout(0.5)(X2)
X3 = keras.layers.add([X1,X2])
X3 = LSTM(NUM_HIDDEN_UNITS)(X3)
X3 = Dropout(0.5)(X3)
output = Dense(vocablen_question,activation = 'sigmoid')(X3)

model = Model(inputs = [input1,input2],outputs = output)
model.compile(optimizer = keras.optimizers.Adam(0.0003),loss = 'binary_crossentropy',metrics = ['accuracy'])
print(model.summary())
model.fit([exp_intseq,questions_intseq],answers_final_form,batch_size = 32,validation_split = 0.15,epochs = 40)