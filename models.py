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

def rnn4(data,units, reg = 0.00, dropout_rate = 0.5, threshold = 1):
    reduced_embedding_matrix = data.reduced_embedding_matrix
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    
    Pooling_layer = GlobalAvgPool1D

    Glove_embedding = Embedding(input_dim = reduced_embedding_matrix.shape[0],output_dim = 300, weights = [reduced_embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Glove_embedding(input_explain)
    X1 = Dropout(0.5)(X1)
    
    X2 = Glove_embedding(input_question)
    X2 = Dropout(0.5)(X2)
    
    RNN = Bidirectional(GRU(units, name = 'combined', dropout = dropout_rate, return_sequences = True, kernel_regularizer = l2(reg), recurrent_regularizer = l2(reg)))
    
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
    
    Cosine_similarity = Lambda(cosine_similarity ,name = 'Cosine_similarity')
    
    pos_similarity = Cosine_similarity([combined_rep,pos_ans_rep])
    neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
    neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
    neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])
    
    loss = Lambda(lambda inputs: hinge_loss(inputs,threshold), name = 'loss')([pos_similarity,neg_similarity1])
    #loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])
        
    
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
#    predictions_normalized = Activation('softmax')(predictions)
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = training_model.get_weights()

    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions)

    return training_model,prediction_model,Wsave

def cnn(data,units = 10,dropout_type = None, dropout_rate = 0.5, reg = 0.00, threshold = 1, filter_nums = None):
    Pooling_layer = GlobalAvgPool1D
    if dropout_type == None:
        Dropouter = Dropout
    elif dropout_type == 'word':
        Dropouter = SpatialDropout1D
    else:
        raise Exception('invalid dropout type')
    if filter_nums == None:
        filter_nums = [10,10,10,10,10,10,10]
    f1,f2,f3,f4,f5,f6,f7 = filter_nums
    
    reduced_embedding_matrix = data.reduced_embedding_matrix
    maxlen_explain = data.lengths.maxlen_exp
    maxlen_question = data.lengths.maxlen_question
    


    def get_conv_model(units):
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
    
    conv_model = get_conv_model(units)
    RNN = Bidirectional(GRU(units, 
                            name = 'answer_lstm', 
                            dropout = dropout_rate,
                            recurrent_dropout = 0.0,
                            return_sequences = True,
                            kernel_regularizer = l2(reg),
                            recurrent_regularizer = l2(reg)))
    
    Glove_embedding = Embedding(input_dim = reduced_embedding_matrix.shape[0],output_dim = reduced_embedding_matrix.shape[1], weights = [reduced_embedding_matrix], name = 'glove_embedding')
    Glove_embedding.trainable = False
    
    input_explain = Input((maxlen_explain,) ,name = 'explanation')
    input_question = Input((maxlen_question,), name = 'question')
    X1 = Glove_embedding(input_explain)
    X2 = Glove_embedding(input_question)
    X1 = Dropouter(dropout_rate)(X1)
    X2 = Dropouter(dropout_rate)(X2)
    
    combined = Concatenate(axis = 1)([X1,X2])
    combined = RNN(combined)
    combined_rep = conv_model(combined)
    
    input_pos_ans = Input((23,))
    input_neg_ans1 = Input((23,))
    input_neg_ans2 = Input((23,))
    input_neg_ans3 = Input((23,))
    
    
#    input_pos_ans_printed = Lambda(lambda x: K.tf.Print(x,data = [x],message = 'pos ans :',summarize = 1000))(input_pos_ans)    

    
    pos_ans = Glove_embedding(input_pos_ans)
    neg_ans1 = Glove_embedding(input_neg_ans1)
    neg_ans2 = Glove_embedding(input_neg_ans2)
    neg_ans3 = Glove_embedding(input_neg_ans3)
    
    
    pos_ans = Dropouter(dropout_rate)(pos_ans)
    neg_ans1 = Dropouter(dropout_rate)(neg_ans1)
    neg_ans2 = Dropouter(dropout_rate)(neg_ans2)
    neg_ans3 = Dropouter(dropout_rate)(neg_ans3)
    
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
#    print(K.int_shape(loss))
#    loss = Lambda(lambda x: K.print_tensor(x,message = 'loss :'))(loss)
#    loss = Lambda(lambda x: K.tf.Print(x,data = [x],message = 'loss :',summarize = 1000))(loss)    
    
    
    predictions = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
#    predictions_normalized = Activation('softmax')(predictions)
    
    untrained_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    Wsave = untrained_model.get_weights()
    
    training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = predictions)
    

    return training_model,prediction_model,Wsave



