#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:02:27 2018

@author: liyuan
"""

from loss_functions import hinge_loss, get_cos_similarity
from keras.layers import Input,Bidirectional,Embedding,GRU,LSTM,Reshape,Average,Add,Activation,GlobalAvgPool1D,Lambda,Concatenate,Dropout
from keras import Model
from loss_functions import hinge_loss, _loss_tensor, tf_cos_similarity, get_norm
from helper_functions import plot_loss_history,save_model_formatted
import tensorflow as tf
import config

word2index = config.word2index
embedding_matrix = config.embedding_matrix

def add_lstm(new_hyperparameters = {}, Pooling_layer = GlobalAvgPool1D,maxlen_explain = None, maxlen_question = None):
        
    hyperparameters = config.hyperparameters
    hyperparameters.update(new_hyperparameters)
    
    max_sentences = 27
    max_sentence_length = 86
    
    num_hidden_units = hyperparameters['num_hidden_units']
    dropout_rate     = hyperparameters['dropout_rate']
    learning_rate    = hyperparameters['learning_rate']
    optimizer        = hyperparameters['optimizer']
    
    RNN = Bidirectional(GRU(num_hidden_units, name = 'answer_lstm', dropout = dropout_rate, recurrent_dropout = 0.1, return_sequences = False))
    Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    Reshaper = Reshape([86])
    
    exp_representations = []
    
    input_explain = Input(shape = [27,86])
    input_question = Input([maxlen_question])
    input_pos_ans = Input((23,))
    input_neg_ans1 = Input((23,))
    input_neg_ans2 = Input((23,))
    input_neg_ans3 = Input((23,))    
    
    
    exp_split = Lambda(lambda x: tf.split(x,max_sentences,1))(input_explain)
    for exp in exp_split:
        exp = Reshaper(exp)
        exp = Glove_embedding(exp)
        exp = RNN(exp)
        exp_representations.append(exp)
    exp_representation = Average()(exp_representations)
    question_representation = Glove_embedding(input_question)
    question_representation = RNN(question_representation)
    combined_rep = Add()([exp_representation,question_representation])
    
    pos_ans = Glove_embedding(input_pos_ans)
    neg_ans1 = Glove_embedding(input_neg_ans1)
    neg_ans2 = Glove_embedding(input_neg_ans2)
    neg_ans3 = Glove_embedding(input_neg_ans3)
    
    pos_ans_rep  = RNN(pos_ans)
    neg_ans_rep1  = RNN(neg_ans1)
    neg_ans_rep2  = RNN(neg_ans2)
    neg_ans_rep3  = RNN(neg_ans3)
    
    Cosine_similarity = Lambda(get_cos_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
    prediction = Activation('softmax')(prediction)
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = prediction)

    
    return training_model,prediction_model
    
        
    
        