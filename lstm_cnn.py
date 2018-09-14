#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:54:32 2018

@author: liyuan
"""

from loss_functions import hinge_loss, tf_cos_similarity
from keras.layers import GlobalAvgPool1D, Input, Embedding, Concatenate, Bidirectional, GRU, Dropout, Lambda
from keras import Model
from loss_functions import hinge_loss, _loss_tensor, tf_cos_similarity, get_norm
from helper_functions import plot_loss_history,save_model_formatted

import config

word2index = config.word2index
embedding_matrix = config.embedding_matrix

def lstm_cnn(new_hyperparameters = {}, Pooling_layer = GlobalAvgPool1D,maxlen_explain = None, maxlen_question = None):
    
    hyperparameters = config.hyperparameters
    hyperparameters.update(new_hyperparameters)

    num_hidden_units = hyperparameters['num_hidden_units']
    dropout_rate     = hyperparameters['dropout_rate']
    learning_rate    = hyperparameters['learning_rate']
    optimizer        = hyperparameters['optimizer']   
    
    conv5 = Conv1D(filters = 3,kernel_size = 5,padding = 'same')
    lstm_ans = Bidirectional(GRU(num_hidden_units, name = 'answer_lstm', dropout = dropout_rate,return_sequences = True))

    Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Glove_embedding(input_explain)
    X2 = Glove_embedding(input_question)
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined_rep = Bidirectional(GRU(num_hidden_units, name = 'combined', dropout = dropout_rate, return_sequences = True))(combined)
    
    conv5_output = conv5(combined_rep)
    conv5_output = GlobalMaxPooling1D()(conv5_output)
    
    input_pos_ans = Input((23,))
    input_neg_ans1 = Input((23,))
    input_neg_ans2 = Input((23,))
    input_neg_ans3 = Input((23,))
    
    pos_ans = Glove_embedding(input_pos_ans)
    neg_ans1 = Glove_embedding(input_neg_ans1)
    neg_ans2 = Glove_embedding(input_neg_ans2)
    neg_ans3 = Glove_embedding(input_neg_ans3)
    
    pos_ans_rep  = lstm_ans(pos_ans)
    pos_ans_rep = conv5(pos_ans_rep)
    pos_ans_rep = GlobalMaxPooling1D()(pos_ans_rep)
    
    neg_ans_rep1  = lstm_ans(neg_ans1)
    neg_ans_rep1 = conv5(neg_ans_rep1)
    neg_ans_rep1 = GlobalMaxPooling1D()(neg_ans_rep1)
    
    neg_ans_rep2  = lstm_ans(neg_ans2)
    neg_ans_rep2 = conv5(neg_ans_rep2)
    neg_ans_rep2 = GlobalMaxPooling1D()(neg_ans_rep2)
    
    neg_ans_rep3  = lstm_ans(neg_ans3)
    neg_ans_rep3 = conv5(neg_ans_rep3)
    neg_ans_rep3 = GlobalMaxPooling1D()(neg_ans_rep3)

    
    
    
    
    
    Cosine_similarity = Lambda(tf_cos_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
    prediction = Activation('softmax')(prediction)
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    training_model.compile(optimizer = optimizer,loss = _loss_tensor,metrics = [])
    
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = prediction)
    prediction_model.compile(optimizer = 'adam', loss = lambda y_true,y_pred: y_pred, metrics = [keras.metrics.categorical_accuracy])
    
    return training_model,prediction_model