#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 17:38:14 2018

@author: liyuan
"""

"""
use babi rnn architecture: compute representation for question, expand this to make a matrix the same dimension as the story/explanation matrix (i.e. after converted into embedding vectors), add this repeat matrix with the story/explanation matrix.
"""



import time
start_time = time.time()


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout,Reshape,Add,Lambda,Concatenate,Bidirectional,GRU, GlobalAvgPool1D, GlobalMaxPool1D, Activation, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
import nltk
from load_glove_embeddings import load_glove_embeddings
import tensorflow as tf
import numpy as np

from loss_functions import hinge_loss, _loss_tensor, get_cos_similarity, get_norm

from wtc_utils import preprocess_data,sample_wrong_answers, convert_to_int, convert_to_letter, get_shuffled_indices
import matplotlib.pyplot as plt
from helper_functions import plot_loss_history,save_model_formatted

import os
import socket
import config

word2index = config.word2index
embedding_matrix = config.embedding_matrix

if socket.gethostname() == 'aai-DGX-Station':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


force_load_data = 1
if force_load_data == 1 or 'questions_intseq' not in vars():
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,explain_intseq,lengths,cache = data
    maxlen_question,maxlen_explain,vocablen_question,vocablen_explain = lengths
    answers_intseq = cache['answers_intseq']
    wrong_answers = cache['wrong_answers']
    all_answer_options_intseq = cache['all_answer_options_intseq']
    answers = cache['answers']
    questions = cache['questions']
    
    answers_intseq2_val = sample_wrong_answers(wrong_answers)
    num_examples = questions_intseq.shape[0]


    # 1463,100,100 split
    num_train = 1363
    num_val = 150
    num_test = 150
    
    
    train_indices,val_indices,test_indices = get_shuffled_indices(num_examples, proportions = [num_train,num_val,num_test])
    
    
    
    
    
    
print("total time taken to load data and embeddings is {:.2f} seconds".format(time.time() - start_time))

#%% keras model

NUM_HIDDEN_UNITS = 10

dropout_rate = 0.5


RNN = Bidirectional(GRU(NUM_HIDDEN_UNITS, name = 'combined', dropout = dropout_rate, return_sequences = True))
Pooling_layer = GlobalAvgPool1D

Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
Glove_embedding.trainable = False

input_explain = Input((maxlen_explain,) ,name = 'explanation')
input_question = Input((maxlen_question,), name = 'question')
X1 = Glove_embedding(input_explain)
X1 = Dropout(0.5)(X1)

X2 = Glove_embedding(input_question)
X2 = Dropout(0.5)(X2)
X2 = Bidirectional(GRU(150, name = 'combined', dropout = dropout_rate, return_sequences = True))(X2)
X2 = Pooling_layer()(X2)
X2 = RepeatVector(maxlen_explain)(X2)

merged = Add()([X1,X2])
combined_rep = Bidirectional(GRU(NUM_HIDDEN_UNITS, name = 'combined', dropout = dropout_rate, return_sequences = True))(merged)
combined_rep = Pooling_layer()(combined_rep)



input_pos_ans = Input((23,))
input_neg_ans1 = Input((23,))
input_neg_ans2 = Input((23,))
input_neg_ans3 = Input((23,))

pos_ans = Glove_embedding(input_pos_ans)
neg_ans1 = Glove_embedding(input_neg_ans1)
neg_ans2 = Glove_embedding(input_neg_ans2)
neg_ans3 = Glove_embedding(input_neg_ans3)

pos_ans = Dropout(0.5)(pos_ans)
neg_ans1 = Dropout(0.5)(neg_ans1)
neg_ans2 = Dropout(0.5)(neg_ans2)
neg_ans3 = Dropout(0.5)(neg_ans3)

pos_ans_rep = Pooling_layer()(RNN(pos_ans))
neg_ans_rep1 = Pooling_layer()(RNN(neg_ans1))
neg_ans_rep2 = Pooling_layer()(RNN(neg_ans2))
neg_ans_rep3 = Pooling_layer()(RNN(neg_ans3))

Cosine_similarity = Lambda(get_cos_similarity ,name = 'Cosine_similarity')

pos_similarity = Cosine_similarity([combined_rep,pos_ans_rep])
neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])


def hinge_loss(inputs):
    similarity1,similarity2 = inputs
#    print(similarity1,similarity2)
    hinge_loss = similarity1 - similarity2 - 1.5
    hinge_loss = -hinge_loss
    loss = K.maximum(0.0,hinge_loss)
    return loss

loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
#loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])

prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
prediction = Activation('softmax')(prediction)

reset_losses = 1
if reset_losses or 'val_loss' not in vars():
    val_loss = np.array([]) 
    training_loss = np.array([])

#%% training
num_iter = 10
LEARNING_RATE = 0.00001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)

training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
print(training_model.summary())

dummy_labels_train = np.array([None]*num_train).reshape(num_train,1)
dummy_labels_val = np.array([None]*num_val).reshape(num_val,1)

history_cache = dict()

for i in range(num_iter):
    print('running iteration {}...'.format(i+1))
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
    X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices],answers_intseq2_val[val_indices]]
    history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = 1024,epochs = 5)
    history_cache[i] = history.history
    val_loss = np.append(val_loss,history.history['val_loss'])
    training_loss = np.append(training_loss,history.history['loss'])

save_plot = 0
titlestr = 'wtc_rnn4_avgpool_'+ str(NUM_HIDDEN_UNITS)
plot_loss_history(training_loss,val_loss,save_image = save_plot,title = titlestr)

    


#%% predict


make_predictions = 1
if make_predictions == 1:
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = prediction)
    prediction_model.compile(optimizer = 'adam', loss = lambda y_true,y_pred: y_pred, metrics = [keras.metrics.categorical_accuracy])
    
    
    all_answer_options_intseq = np.array(all_answer_options_intseq)
    
    input1 = explain_intseq
    input2 = questions_intseq
    input3 = all_answer_options_intseq[:,0,:]
    input4 = all_answer_options_intseq[:,1,:]
    input5 = all_answer_options_intseq[:,2,:]
    input6 = all_answer_options_intseq[:,3,:]  
        
    predict_output = prediction_model.predict([input1,input2,input3,input4,input5,input6],batch_size = 1)
    predicted_ans = np.argmax(predict_output,axis = 1)
    print(predict_output)
    print(predicted_ans)
    
    
    int_ans = np.array([convert_to_int(letter) for letter,ans in answers])
    train_acc = np.mean(predicted_ans[train_indices] == int_ans[train_indices])
    val_acc = np.mean(predicted_ans[val_indices] == int_ans[val_indices])
    test_acc = np.mean(predicted_ans[test_indices] == int_ans[test_indices])
    
    print('train,val,test accuracies: {:.2f}/{:.2f}/{:.2f}'.format(train_acc,val_acc,test_acc))

#%% save model


save_model = 0
if save_model == 1:
    save_model_formatted(prediction_model,NUM_HIDDEN_UNITS)