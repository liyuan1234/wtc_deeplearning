#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:50:04 2018

@author: liyuan
"""

import time
start_time = time.time()

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout,Reshape,Add,Lambda
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
import nltk
from load_glove_embeddings import load_glove_embeddings
import tensorflow as tf
import numpy as np

from loss_functions import hinge_loss, _loss_tensor, get_cosine_similarity, get_norm

from wtc_utils import preprocess_data,sample_wrong_answers

load_embeddings = 1
if load_embeddings == 1:
    word2index, embedding_matrix = load_glove_embeddings('./embeddings/glove.6B.300d.txt', embedding_dim=300)

load_data = 1
if load_data == 1:
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,explain_intseq,lengths,cache = data
    maxlen_question,maxlen_explain,vocablen_question,vocablen_explain = lengths
    questions_vocab_idx,questions_vocab,questions,answers,answers_intseq,explain_vocab,explain_vocab_dict,explain_tokenized,all_answer_options_with_questions,all_answer_options,all_answer_options_intseq,wrong_answers = cache
    
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    num_examples = questions_intseq.shape[0]
    
print("--- {:.2f} seconds ---".format(time.time() - start_time))

#%% keras model

NUM_HIDDEN_UNITS = 500

Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix])
Glove_embedding.trainable = False

input1 = Input((maxlen_explain,))
X1 = Glove_embedding(input1)
X1 = Dropout(0.5)(X1)
output1 = LSTM(NUM_HIDDEN_UNITS)(X1)

input2 = Input((maxlen_question,))

X2 = Glove_embedding(input2)
X2 = Dropout(0.5)(X2)

rep_explain_ques = Add()([output1,output2])

lstm_ans = LSTM(NUM_HIDDEN_UNITS)

input3 = Input((23,))
input4 = Input((23,))

pos_ans = Glove_embedding(input3)
neg_ans = Glove_embedding(input4)

pos_ans_rep = lstm_ans(pos_ans)
neg_ans_rep = lstm_ans(neg_ans)

similarity1 = Lambda(get_cosine_similarity)([rep_explain_ques,pos_ans_rep])
similarity2 = Lambda(get_cosine_similarity)([rep_explain_ques,neg_ans_rep])

loss = Lambda(hinge_loss)([similarity1,similarity2])

#%% training
LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)
#OPTIMIZER = keras.optimizers.RMSprop(lr = 0.0001)













model = Model(inputs = [input1,input2,input3,input4],outputs = loss)
model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
#print(model.summary())

dummy_labels = np.array([None]*num_examples).reshape(num_examples,1)

model.fit(x = [explain_intseq,questions_intseq,answers_intseq,answers_intseq2],y = dummy_labels,batch_size = 256,validation_split = 0.3,epochs = 20)


#%% more training
    
num_iter = 50

for i in range(num_iter):
    print('running iteration {}...'.format(i))
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    
    LEARNING_RATE = 0.00001
    OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)
    #OPTIMIZER = keras.optimizers.RMSprop(lr = 0.0001)
    
    model = Model(inputs = [input1,input2,input3,input4],outputs = loss)
    model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
    model.fit(x = [explain_intseq,questions_intseq,answers_intseq,answers_intseq2],y = dummy_labels,batch_size = 256,validation_split = 0.3,epochs = 10)
#%% save model
    
save_model = 0
if save_model == 1:
    model.save('./saved_models/rnn3.h5py')    
