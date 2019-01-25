#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:16:38 2018

@author: liyuan
"""
from loss_functions import hinge_loss, cosine_similarity
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,GRU,Bidirectional
from keras.layers import Dense,Input,Reshape,Lambda,Add,Activation,Concatenate,Conv2D
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D,GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import Dropout, SpatialDropout1D
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

def model(data,units, model_flag = 'normal', reg = 0.00, dropout_rate = 0.5, threshold = 1, rnn_layers = 1,filter_nums = None):
    
    '''set up'''
    reduced_embedding_matrix = data.reduced_embedding_matrix
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    
    if model_flag == 'normal':
        use_cnn = 0
        model_flag = model_flag + str(rnn_layers)
    elif model_flag == 'cnn':
        use_cnn = 1
        if filter_nums == None:
            filter_nums = [10,10,10,10,10,10,10]
    else:
        raise('invalid model_flag.')
    
    ''' define some layers'''
    RNN = get_rnn(units = units,
                  layers = rnn_layers,
                  dropout = dropout_rate,
                  reg = reg,
                  use_cnn = use_cnn,
                  filter_nums = filter_nums)         



    Cosine_similarity = Lambda(cosine_similarity ,name = 'Cosine_similarity')
    Hinge_loss = Lambda(lambda inputs: hinge_loss(inputs,threshold), name = 'loss')
    
    '''define model'''
    e_input = Input((maxlen_explain,) ,name = 'explanation')
    q_input = Input((maxlen_question,), name = 'question')
    e = Glove_embedding(e_input)
    q = Glove_embedding(q_input)
    e = Dropout(0.5)(e)    
    q = Dropout(0.5)(q)
    eq = Concatenate(axis = 1)([e,q])
    eq = RNN(eq)
    pos_ans_input = Input((23,))
    neg_ans1_input = Input((23,))
    neg_ans2_input = Input((23,))
    neg_ans3_input = Input((23,))
    pos_ans = Glove_embedding(pos_ans_input)
    neg_ans1 = Glove_embedding(neg_ans1_input)
    neg_ans2 = Glove_embedding(neg_ans2_input)
    neg_ans3 = Glove_embedding(neg_ans3_input)
    pos_ans = Dropout(0.5)(pos_ans)
    neg_ans1 = Dropout(0.5)(neg_ans1)
    neg_ans2 = Dropout(0.5)(neg_ans2)
    neg_ans3 = Dropout(0.5)(neg_ans3)
    pos_ans = RNN(pos_ans)
    neg_ans1 = RNN(neg_ans1)
    neg_ans2 = RNN(neg_ans2)
    neg_ans3 = RNN(neg_ans3)
    pos_similarity = Cosine_similarity([eq,pos_ans])
    neg_similarity1 = Cosine_similarity([eq,neg_ans1])
    neg_similarity2 = Cosine_similarity([eq,neg_ans2])
    neg_similarity3 = Cosine_similarity([eq,neg_ans3])
    loss = Hinge_loss([pos_similarity,neg_similarity1])
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
    
    ''' define training_model and prediction_model'''
    training_model = Model(inputs = [e_input,q_input,pos_ans_input,neg_ans1_input],outputs = loss)
    Wsave = training_model.get_weights()
    prediction_model = Model(inputs = [e_input,q_input,pos_ans_input,neg_ans1_input,neg_ans2_input,neg_ans3_input],outputs = predictions)

    return training_model,prediction_model,Wsave, model_flag

def get_rnn(units, layers, dropout, reg, use_cnn, filter_nums):
    '''
    RNN is the stack of processing that transforms from input to representation
    consists of a stack of RNN and can use a cnn at the end
    if use_cnn == 1, uses cnn to 'pool' at the end
    else uses average pooling to pool 
    '''
    
    embedding_input = Input([None,300])
    rep = embedding_input
    for i in range(layers):
        rep = Bidirectional(GRU(units, name = 'RNN'+str(i), dropout = dropout, return_sequences = True, kernel_regularizer = l2(reg), recurrent_regularizer = l2(reg)))(rep)
    if not use_cnn:
        output = GlobalAvgPool1D()(rep)
    if use_cnn:
        cnn = get_conv_model(units, filter_nums)
        output = cnn(rep)
    
    RNN = Model(inputs = embedding_input, outputs = output)
    return RNN
    
def get_conv_model(units, filter_nums = None):
    f1,f2,f3,f4,f5,f6,f7 = filter_nums
    biLSTM_units = units*2
    inputer = Input([None,biLSTM_units])
    input_representation = Reshape([-1,biLSTM_units,1])(inputer)
    c1 = Conv2D(filters = f1, kernel_size = [2,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)     
    c1 = GlobalMaxPooling2D()(c1)        
    c2 = Conv2D(filters = f2, kernel_size = [2,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c2 = GlobalMaxPooling2D()(c2)
    c3 = Conv2D(filters = f3, kernel_size = [3,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c3 = GlobalMaxPooling2D()(c3)
    c4 = Conv2D(filters = f4, kernel_size = [4,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c4 = GlobalMaxPooling2D()(c4)
    c5 = Conv2D(filters = f5, kernel_size = [5,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c5 = GlobalMaxPooling2D()(c5)
    c6 = Conv2D(filters = f6, kernel_size = [6,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c6 = GlobalMaxPooling2D()(c6)
    c7 = Conv2D(filters = f7, kernel_size = [7,biLSTM_units], padding = 'valid', activation = 'tanh', data_format = 'channels_last')(input_representation)
    c7 = GlobalMaxPooling2D()(c7)
#    conv8_output = Conv2D(filters = 2, kernel_size = 10, padding = 'same', activation = 'tanh')(input_representation)
#    conv8_output = GlobalMaxPooling1D()(conv8_output)
#    conv9_output = Conv2D(filters = 2, kernel_size = 15, padding = 'same', activation = 'tanh')(input_representation)
#    conv9_output = GlobalMaxPooling1D()(conv9_output)    
    conv_output = Concatenate(axis = 1)([c1,c2,c3,c4,c5,c6,c7])
    conv_model = Model(inputs = inputer,outputs = conv_output)
    return conv_model    





