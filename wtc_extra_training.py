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


LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)


file_path = './saved_models/rnn3.h5py'
model = keras.models.load_model(file_path)

model = Model(inputs = [input1,input2],outputs = output)
model.compile(optimizer = OPTIMIZER,loss = 'categorical_crossentropy',metrics = ['accuracy'])
print(model.summary())
model.fit([exp_intseq,questions_intseq],answers_final_form,batch_size = 64,validation_split = 0.15,epochs = 10)

save_model = 1
if save_model == 1:
    model.save('./saved_models/rnn3.h5py')