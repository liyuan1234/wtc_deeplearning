#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:16:43 2018

@author: liyuan
"""
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout
from keras.models import Model
import keras
from wtc_utils import preprocess_data


#%%
load_data = 0
if load_data:
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,exp_intseq,lengths,cache = data
    maxlen_question,maxlen_exp,vocablen_question,vocablen_exp = lengths
    questions_vocab_idx,questions_vocab,questions,answers,answers_intseq,exp_vocab,exp_vocab_dict,exp_tokenized = cache


#%%
LEARNING_RATE = 0.0001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)
BATCH_SIZE = 2000
EPOCHS = 200


file_path = './saved_models/rnn3.h5py'
model = keras.models.load_model(file_path)

model.compile(optimizer = OPTIMIZER,loss = 'binary_crossentropy',metrics = ['accuracy'])
#print(model.summary())
model.fit([exp_intseq,questions_intseq],answers_final_form,batch_size = BATCH_SIZE,validation_split = 0.15,epochs = EPOCHS)

save_model = 1
if save_model == 1:
    model.save('./saved_models/rnn3.h5py')
    
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