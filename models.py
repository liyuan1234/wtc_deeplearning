#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:16:38 2018

@author: liyuan
"""
from loss_functions import hinge_loss, get_cosine_similarity
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout,Reshape,Add,Lambda,Concatenate,Bidirectional,GRU, GlobalAvgPool1D, GlobalMaxPool1D, Activation, Conv1D,GlobalMaxPooling1D
from keras.models import Model


def rnn4(data,num_hidden_units):
    embedding_matrix = data.embedding_matrix
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    
    Pooling_layer = GlobalAvgPool1D
    dropout_rate = 0.5
    
    Glove_embedding = Embedding(input_dim = embedding_matrix.shape[0],output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Glove_embedding(input_explain)
    X1 = Dropout(0.5)(X1)
    
    X2 = Glove_embedding(input_question)
    X2 = Dropout(0.5)(X2)
    
    RNN = Bidirectional(GRU(num_hidden_units, name = 'combined', dropout = dropout_rate, return_sequences = True))
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined_rep = RNN(combined)
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
    
    Cosine_similarity = Lambda(get_cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
    predictions_normalized = Activation('softmax')(prediction)
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = training_model.get_weights()

    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions_normalized)
    
    title = 'rnn4_'+str(num_hidden_units)+'units'

    return training_model,prediction_model,Wsave,title


def cnn(data,num_hidden_units = 10):
    embedding_matrix = data.embedding_matrix
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    
    Pooling_layer = GlobalAvgPool1D
    dropout_rate = 0.5

    def get_conv_model(num_hidden_units):
        input_representation = Input(shape = [None,num_hidden_units*2])
        conv2_output = Conv1D(filters = 2, kernel_size = 2, padding = 'same', activation = 'tanh')(input_representation)
        conv2_output = GlobalMaxPooling1D()(conv2_output)
        conv3_output = Conv1D(filters = 2, kernel_size = 3, padding = 'same', activation = 'tanh')(input_representation)
        conv3_output = GlobalMaxPooling1D()(conv3_output)
        conv4_output = Conv1D(filters = 2, kernel_size = 4, padding = 'same', activation = 'tanh')(input_representation)
        conv4_output = GlobalMaxPooling1D()(conv4_output)
        conv5_output = Conv1D(filters = 2, kernel_size = 5, padding = 'same', activation = 'tanh')(input_representation)
        conv5_output = GlobalMaxPooling1D()(conv5_output)
        conv6_output = Conv1D(filters = 2, kernel_size = 6, padding = 'same', activation = 'tanh')(input_representation)
        conv6_output = GlobalMaxPooling1D()(conv6_output)
        conv7_output = Conv1D(filters = 2, kernel_size = 7, padding = 'same', activation = 'tanh')(input_representation)
        conv7_output = GlobalMaxPooling1D()(conv7_output)
    #    conv8_output = Conv1D(filters = 2, kernel_size = 10, padding = 'same', activation = 'tanh')(input_representation)
    #    conv8_output = GlobalMaxPooling1D()(conv8_output)
    #    conv9_output = Conv1D(filters = 2, kernel_size = 15, padding = 'same', activation = 'tanh')(input_representation)
    #    conv9_output = GlobalMaxPooling1D()(conv9_output)    
        conv_output = Concatenate(axis = 1)([conv2_output,conv3_output,conv4_output,conv5_output,conv6_output,conv7_output])
        conv_model = Model(inputs = input_representation,outputs = conv_output)
        return conv_model    
    
    conv_model = get_conv_model(num_hidden_units)
    RNN = Bidirectional(GRU(num_hidden_units, name = 'answer_lstm', dropout = 0.5,recurrent_dropout = 0.2,return_sequences = True))
    
    Glove_embedding = Embedding(input_dim = embedding_matrix.shape[0],output_dim = embedding_matrix.shape[1], weights = [embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Glove_embedding(input_explain)
    X2 = Glove_embedding(input_question)
    X1 = Dropout(0.5)(X1)
    X2 = Dropout(0.5)(X2)
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined = RNN(combined)
    combined_rep = conv_model(combined)
    
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
    
    pos_ans  = RNN(pos_ans)
    pos_ans_rep = conv_model(pos_ans)
    
    neg_ans1  = RNN(neg_ans1)
    neg_ans_rep1 = conv_model(neg_ans1)
    
    neg_ans2  = RNN(neg_ans2)
    neg_ans_rep2 = conv_model(neg_ans2)
    
    neg_ans3  = RNN(neg_ans3)
    neg_ans_rep3 = conv_model(neg_ans3)
    
    Cosine_similarity = Lambda(get_cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
    predictions_normalized = Activation('softmax')(predictions)
    
    untrained_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = untrained_model.get_weights()
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions_normalized)
    
    title = 'cnn_'+str(num_hidden_units)+'units'

    return training_model,prediction_model,Wsave,title

