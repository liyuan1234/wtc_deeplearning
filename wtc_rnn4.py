#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:50:04 2018

@author: liyuan
"""

"""
rnn4 architecture:
    calculate 2 representations respectively for the question and explanation (pass question/explanation through an LSTM, use different LSTM for question and explanation), then add these representations together to get for example a length 100 vector (the combined representation). Calculate representations for each answer, get cosine similarity between the question/explanation representation and answer representation, choose 

"""
import time
start_time = time.time()


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout,Reshape,Add,Lambda,Concatenate,Bidirectional,GRU, GlobalAvgPool1D, GlobalMaxPool1D, Activation
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
import nltk
from load_glove_embeddings import load_glove_embeddings
import tensorflow as tf
import numpy as np

from loss_functions import hinge_loss, _loss_tensor, get_cosine_similarity, get_norm

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
Pooling_layer = GlobalAvgPool1D
dropout_rate = 0.5

Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
Glove_embedding.trainable = False

input_explain = Input((maxlen_explain,) ,name = 'explanation')
input_question = Input((maxlen_question,), name = 'question')
X1 = Glove_embedding(input_explain)
X2 = Glove_embedding(input_question)

combined = Concatenate(axis = 1)([X1,X2])
combined_rep = Bidirectional(GRU(NUM_HIDDEN_UNITS, name = 'combined', dropout = dropout_rate, return_sequences = True))(combined)
combined_rep = Pooling_layer()(combined_rep)



lstm_ans = Bidirectional(GRU(NUM_HIDDEN_UNITS, name = 'answer_lstm', dropout = dropout_rate,return_sequences = True))

input_pos_ans = Input((23,))
input_neg_ans1 = Input((23,))
input_neg_ans2 = Input((23,))
input_neg_ans3 = Input((23,))

pos_ans = Glove_embedding(input_pos_ans)
neg_ans1 = Glove_embedding(input_neg_ans1)
neg_ans2 = Glove_embedding(input_neg_ans2)
neg_ans3 = Glove_embedding(input_neg_ans3)

pos_ans_rep = Pooling_layer()(lstm_ans(pos_ans))
neg_ans_rep1 = Pooling_layer()(lstm_ans(neg_ans1))
neg_ans_rep2 = Pooling_layer()(lstm_ans(neg_ans2))
neg_ans_rep3 = Pooling_layer()(lstm_ans(neg_ans3))

Cosine_similarity = Lambda(get_cosine_similarity ,name = 'Cosine_similarity')

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



#%% training
num_iter = 20
LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)

training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
print(training_model.summary())

dummy_labels_train = np.array([None]*num_train).reshape(num_train,1)
dummy_labels_val = np.array([None]*num_val).reshape(num_val,1)

history_cache = dict()

reset_losses = 0
if reset_losses or 'val_loss' not in vars():
    val_loss = np.array([]) 
if reset_losses or 'training_loss' not in vars():
    training_loss = np.array([])
for i in range(num_iter):
    print('running iteration {}...'.format(i+1))
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
    X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices],answers_intseq2_val[val_indices]]
    history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = 128,epochs = 5)
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
    
    indices = train_indices # test_indices or training_indices
    temp1 = explain_intseq[indices]
    temp2 = questions_intseq[indices]
    temp3 = all_answer_options_intseq[indices,0,:]
    temp4 = all_answer_options_intseq[indices,1,:]
    temp5 = all_answer_options_intseq[indices,2,:]
    temp6 = all_answer_options_intseq[indices,3,:]  
    
    #model2.predict([temp1,temp2,temp3,temp4,temp5,temp6])
    
    predict_output = prediction_model.predict([temp1,temp2,temp3,temp4,temp5,temp6],batch_size = 1)
    predicted_ans = np.argmax(predict_output,axis = 1)
    print(predict_output)
    print(predicted_ans)
    
    
    int_ans = np.array([convert_to_int(letter) for letter,ans in answers])
    print(np.mean(predicted_ans == int_ans[indices]))
    
#    prediction_model.evaluate([temp1,temp2,temp3,temp4,temp5,temp6],correct_ans,verbose = False)

#%% save model


save_model = 0
if save_model == 1:
    save_model_formatted(prediction_model,NUM_HIDDEN_UNITS)
