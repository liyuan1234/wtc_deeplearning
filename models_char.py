#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:08:07 2018

@author: liyuan
"""

from loss_functions import hinge_loss, cosine_similarity
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,GRU,Bidirectional
from keras.layers import Dense,Input,Reshape,Lambda,Add,Activation,Concatenate,Conv1D,Conv2D
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D,GlobalMaxPooling1D,GlobalMaxPooling2D
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import TimeDistributed
from keras.models import Model


import keras
from keras.engine.topology import Layer
from keras import layers
from keras import Model
import keras.backend as K
import tensorflow as tf

from Char_embedding_layer import cnn_layer, cnn_lstm_layer


class WordEmbeddings(Layer):
    def __init__(self, voc_size, embed_dim):
        self.voc_size = voc_size
        self.embed_dim = embed_dim
        super().__init__()
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
                shape = [self.voc_size,self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'word_embeddings',
                trainable = False)
        #trainable = False
        
        self.built = True
        
    def call(self, inputs):
        if not len(inputs) == 3 and isinstance(inputs,list):
            raise Exception('input must be [indices, new_embeddings, qs] format')
        
        indices, emb_update, qs = inputs
        self.kernel = K.tf.scatter_add(self.kernel,indices, emb_update)
        vs = K.gather(self.kernel, qs)
        
        return vs
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape1, shape2, shape3 = input_shape
        assert isinstance(shape3, tuple)
        return shape3 + (self.embed_dim,)


def model(data,units = 10, units_char = 10, embedding_dim = 15, threshold = 1, model_flag = 'cnn_lstm',filter_counts = None, dropout_rate = 0.5):
    from Char_embedding_layer import cnn_layer, cnn_lstm_layer
    
    max_char_in_word = data.lengths.max_char_in_word
    num_char = data.lengths.char2index_length
    maxlen_question = data.lengths.maxlen_question
    maxlen_exp = data.lengths.maxlen_exp    
    maxlen_answer = data.lengths.maxlen_answer
    input_shape_exp = [maxlen_exp,max_char_in_word]
    input_shape_question = [maxlen_question,max_char_in_word]
    input_shape_answer = [maxlen_answer,max_char_in_word]
    
    
    RNN = Bidirectional(GRU(units, name = 'answer_lstm', dropout = dropout_rate, recurrent_dropout = 0.0,return_sequences = False))
    
    
    if filter_counts == None:
        filter_counts = [10,10,10,10,10,10]
    
    
    if model_flag == 'cnn':
        Char_Embedding = cnn_layer(num_char, embedding_dim,filter_counts = filter_counts)
    elif model_flag == 'cnn_lstm':
        Char_Embedding = cnn_lstm_layer(num_char,embedding_dim,units_char,filter_counts = filter_counts)
    else:
        raise Exception('invalid flag')
    
    input_explain = Input(input_shape_exp ,name = 'explanation')
    input_question = Input(input_shape_question, name = 'question')
    X1 = Char_Embedding(input_explain)
    X2 = Char_Embedding(input_question)
    X1 = Dropout(0.5)(X1)
    X2 = Dropout(0.5)(X2)
    
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined_rep = RNN(combined)
    
    input_pos_ans = Input(input_shape_answer)
    input_neg_ans1 = Input(input_shape_answer)
    input_neg_ans2 = Input(input_shape_answer)
    input_neg_ans3 = Input(input_shape_answer)
    
    pos_ans = Char_Embedding(input_pos_ans)
    neg_ans1 = Char_Embedding(input_neg_ans1)
    neg_ans2 = Char_Embedding(input_neg_ans2)
    neg_ans3 = Char_Embedding(input_neg_ans3)
    
    pos_ans = Dropout(0.5)(pos_ans)
    neg_ans1 = Dropout(0.5)(neg_ans1)
    neg_ans2 = Dropout(0.5)(neg_ans2)
    neg_ans3 = Dropout(0.5)(neg_ans3)
    
    pos_ans_rep  = RNN(pos_ans)
    
    neg_ans_rep1  = RNN(neg_ans1)
    
    neg_ans_rep2  = RNN(neg_ans2)
    
    neg_ans_rep3  = RNN(neg_ans3)
    
    Cosine_similarity = Lambda(cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(lambda inputs: hinge_loss(inputs,threshold), name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
#    predictions_normalized = Activation('softmax')(predictions)
    
    untrained_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = untrained_model.get_weights()
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions)
    
    return training_model,prediction_model,Wsave, model_flag


def lstm_cnn(data,units = 10, threshold = 1):
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    maxlen_answer = data.lengths.maxlen_answer
    
    Pooling_layer = GlobalAvgPool1D
    dropout_rate = 0.5

    def get_conv_model(units):
        input_representation = Input(shape = [None,units*2])
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
    
    conv_model = get_conv_model(units)
    RNN = Bidirectional(GRU(units, name = 'answer_lstm', dropout = 0.5,recurrent_dropout = 0.2,return_sequences = True))
    
    Char_embedding = Embedding(data.lengths.char2index_length,200)
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Char_embedding(input_explain)
    X2 = Char_embedding(input_question)
    X1 = Dropout(0.5)(X1)
    X2 = Dropout(0.5)(X2)
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined = RNN(combined)
    combined_rep = conv_model(combined)
    
    input_pos_ans = Input((maxlen_answer,))
    input_neg_ans1 = Input((maxlen_answer,))
    input_neg_ans2 = Input((maxlen_answer,))
    input_neg_ans3 = Input((maxlen_answer,))
    
    pos_ans = Char_embedding(input_pos_ans)
    neg_ans1 = Char_embedding(input_neg_ans1)
    neg_ans2 = Char_embedding(input_neg_ans2)
    neg_ans3 = Char_embedding(input_neg_ans3)
    
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
    
    Cosine_similarity = Lambda(cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(lambda inputs: hinge_loss(inputs,threshold), name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
#    predictions_normalized = Activation('softmax')(predictions)
    
    untrained_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = untrained_model.get_weights()
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions)
    
    return training_model,prediction_model,Wsave


def cnn(data,units = 10, threshold = 1):
    
    max_char_in_word = data.lengths.max_char_in_word
    num_char = data.lengths.char2index_length
    embedding_dim = 100
    
    def get_conv_model(embedding_dim):
        input_representation = Input([max_char_in_word,embedding_dim,1])
        conv2_output = Conv2D(filters = 3, kernel_size = [2,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
        conv2_output = GlobalMaxPooling2D()(conv2_output)
        conv3_output = Conv2D(filters = 4, kernel_size = [3,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
        conv3_output = GlobalMaxPooling2D()(conv3_output)
        conv4_output = Conv2D(filters = 5, kernel_size = [4,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
        conv4_output = GlobalMaxPooling2D()(conv4_output)
        conv5_output = Conv2D(filters = 2, kernel_size = [5,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
        conv5_output = GlobalMaxPooling2D()(conv5_output)
#        conv6_output = Conv2D(filters = 2, kernel_size = [6,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
#        conv6_output = GlobalMaxPooling2D()(conv6_output)
#        conv7_output = Conv2D(filters = 2, kernel_size = [7,embedding_dim], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
#        conv7_output = GlobalMaxPooling2D()(conv7_output)
    #    conv8_output = Conv2D(filters = 2, kernel_size = 10, padding = 'same', activation = 'tanh')(input_representation)
    #    conv8_output = GlobalMaxPooling1D()(conv8_output)
    #    conv9_output = Conv2D(filters = 2, kernel_size = 15, padding = 'same', activation = 'tanh')(input_representation)
    #    conv9_output = GlobalMaxPooling1D()(conv9_output)    
        conv_output = Concatenate(axis = 1)([conv2_output,conv3_output,conv4_output,conv5_output])
        conv_model = Model(inputs = input_representation,outputs = conv_output)
        conv_model.name = 'cnn'
        return conv_model  
    

    RNN = Bidirectional(GRU(units, name = 'answer_lstm', dropout = 0.2,recurrent_dropout = 0.0,return_sequences = False))
    
    
    input_shape = [None,max_char_in_word]
    char_rep_shape = [-1,max_char_in_word,embedding_dim,1]
    
    def get_ce_model():
        
        conv_model = get_conv_model(embedding_dim)
        char_lookup = Embedding(num_char,embedding_dim)
        
        inputer = Input(shape = input_shape)
        char_rep = char_lookup(inputer)
        char_rep = Dropout(0.5)(char_rep)
        char_rep = Reshape(char_rep_shape)(char_rep)
        encoded_sentence = TimeDistributed(conv_model)(char_rep)
        Char_Embedding = Model(inputs = inputer,outputs = encoded_sentence)
        Char_Embedding.name = 'Character_Embedding'
        
        return Char_Embedding
    
    Char_Embedding = get_ce_model()
    input_explain = Input(input_shape ,name = 'explanation')
    input_question = Input(input_shape, name = 'question')
    X1 = Char_Embedding(input_explain)
    X2 = Char_Embedding(input_question)
    X1 = Dropout(0.2)(X1)
    X2 = Dropout(0.2)(X2)
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined_rep = RNN(combined)
    
    input_pos_ans = Input(input_shape)
    input_neg_ans1 = Input(input_shape)
    input_neg_ans2 = Input(input_shape)
    input_neg_ans3 = Input(input_shape)
    
    pos_ans = Char_Embedding(input_pos_ans)
    neg_ans1 = Char_Embedding(input_neg_ans1)
    neg_ans2 = Char_Embedding(input_neg_ans2)
    neg_ans3 = Char_Embedding(input_neg_ans3)
    
    pos_ans = Dropout(0.2)(pos_ans)
    neg_ans1 = Dropout(0.2)(neg_ans1)
    neg_ans2 = Dropout(0.2)(neg_ans2)
    neg_ans3 = Dropout(0.2)(neg_ans3)
    pos_ans_rep  = RNN(pos_ans)
    neg_ans_rep1  = RNN(neg_ans1)
    neg_ans_rep2  = RNN(neg_ans2)
    neg_ans_rep3  = RNN(neg_ans3)
    
    Cosine_similarity = Lambda(cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(lambda inputs: hinge_loss(inputs,threshold), name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
#    predictions_normalized = Activation('softmax')(predictions)
    
    untrained_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = untrained_model.get_weights()
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions)
     
    return training_model,prediction_model,Wsave   
    