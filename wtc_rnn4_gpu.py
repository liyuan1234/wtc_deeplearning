#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:31:40 2018

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

from wtc_utils import preprocess_data, get_cosine_similarity, hinge_loss

#word2index, embedding_matrix = load_glove_embeddings('./word embeddings/glove.6B.50d.txt', embedding_dim=50)

load_data = 1
if load_data:
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,explain_intseq,lengths,cache = data
    maxlen_question,maxlen_explain,vocablen_question,vocablen_explain = lengths
    questions_vocab_idx,questions_vocab,questions,answers,answers_intseq,explain_vocab,explain_vocab_dict,explain_tokenized = cache
    
    num_examples = questions_intseq.shape[0]
    answers_intseq = pad_sequences(answers_intseq)
    answers_intseq2 = np.random.permutation(answers_intseq)
    
    
    


## convert answers to 1,2,3,4
#answers_int = [option for option,_ in answers]
#answers_int = list([convert_to_int(x) for x in answers_int])
#answers_onehot = to_categorical(answers_int)

print("--- {:.2f} seconds ---".format(time.time() - start_time))

#%% keras model

NUM_HIDDEN_UNITS = 100

input1 = Input((maxlen_explain,))
X1 = Embedding(input_dim = vocablen_explain,output_dim = NUM_HIDDEN_UNITS)(input1)
X1 = Dropout(0.5)(X1)
output1 = LSTM(NUM_HIDDEN_UNITS)(X1)

input2 = Input((maxlen_question,))
shared_embedding = Embedding(input_dim = vocablen_question,output_dim = NUM_HIDDEN_UNITS)

X2 = shared_embedding(input2)
X2 = Dropout(0.5)(X2)
output2 = LSTM(NUM_HIDDEN_UNITS)(X2)

rep_explain_ques = Add()([output1,output2])

lstm_ans = LSTM(NUM_HIDDEN_UNITS)

input3 = Input((23,))
input4 = Input((23,))

pos_ans = shared_embedding(input3)
neg_ans = shared_embedding(input4)

pos_ans_rep = lstm_ans(pos_ans)
neg_ans_rep = lstm_ans(neg_ans)

similarity1 = Lambda(get_cosine_similarity)([rep_explain_ques,pos_ans_rep])
similarity2 = Lambda(get_cosine_similarity)([rep_explain_ques,neg_ans_rep])


def hinge_loss(inputs):
    similarity1,similarity2 = inputs
#    print(similarity1,similarity2)
    hinge_loss = similarity1 - similarity2 - 2.5
    hinge_loss = -hinge_loss
    loss = K.maximum(0.0,hinge_loss)
    return loss


loss = Lambda(hinge_loss)([similarity1,similarity2])




#%%

def _loss_tensor(y_true,y_pred):
    return y_pred

#% training
LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)
#OPTIMIZER = keras.optimizers.RMSprop(lr = 0.0001)

model = Model(inputs = [input1,input2,input3,input4],outputs = loss)
model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
#print(model.summary())

dummy_labels = np.array([None]*num_examples).reshape(num_examples,1)

model.fit(x = [explain_intseq,questions_intseq,answers_intseq,answers_intseq2],y = dummy_labels,batch_size = 256,validation_split = 0.3,epochs = 10)


    
#%% more training
    
num_iter = 5

for i in range(num_iter):
    answers_intseq2 = np.random.permutation(answers_intseq)
    
    LEARNING_RATE = 0.001
    OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)
    #OPTIMIZER = keras.optimizers.RMSprop(lr = 0.0001)
    
    model = Model(inputs = [input1,input2,input3,input4],outputs = loss)
    model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
    model.fit(x = [explain_intseq,questions_intseq,answers_intseq,answers_intseq2],y = dummy_labels,batch_size = 256,validation_split = 0.3,epochs = 10)


#%% save model
    
save_model = 1
if save_model == 1:
    model.save('./saved_models/rnn4.h5py')    