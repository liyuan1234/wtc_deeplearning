#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:35:28 2018

@author: liyuan
"""
"""
cnn_layer and cnn_lstm_layer are "character embedding" layers that effectively perform word embedding. Word embeddings are inferred from the characters in the words. Input (excluding batch dimension) should be a 2d array of words x characters.
"""


import keras
from keras.engine.topology import Layer
from keras import layers
from keras import Model
import keras.backend as K
import tensorflow as tf

class cnn_layer(Layer):
    """
    Arguments
    num_vocab: number of unique characters
    embed_dim: character embedding dimension 
    filter_counts: list of integers specifying the number of filters for each window size. 
    (If a certain window size is not required, the corresponding element needs to be 0. e.g. to specify
     one filter of size 3, filter_counts needs to be [0,0,1])
    
    Details:
    cnn_layer first converts character sequence to character embeddings, then performs convolution over 
    the character embeddings for each word. The character embeddings for each word is a 2d array, say
    17x15. A convolution filter is then applied to this array, iterating through the characters 
    (convolution filter will be nx15 dimension, covering the character embedding dimension). This 
    reduces the embedding dimension from 15 to 1, so you get 15x1 array. Then max pooling is applied, 
    producing a single convolutional output for each word (for one filter). Because many filters are 
    used, the embedding for each word is a vector where each element corresponds to the convolution 
    output of one filter.
    
    Example usage:
    myinput = Input(100,20)
    embed = cnn_layer(num_vocab = 40, embed_dim = 15, filter_counts = [10,10,10,10,10,10])(myinput)
    output = LSTM(10)(embed)
    
    # myinput will be a tensor with shape (None,100,20)
    # embed will be a tensor with shape (None,100,60)
    # output embeding dimension is sum of filter_counts
    
    """
    def __init__(self, num_vocab, embed_dim, filter_counts = None, **kwargs):
        if filter_counts == None:
            raise Exception('must specify filter counts!')
        
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.filter_counts = filter_counts
        self.conv_output_len = sum(self.filter_counts)        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        #add kernel for embedding
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings')
        
        # add kernel for filters
        filter_counts = self.filter_counts
        for i in range(len(filter_counts)):
            if filter_counts[i] != 0:
                name = 'kernel'+str(i+1)
                weights = self.add_weight(
                        shape = [i+1,self.embed_dim,1,filter_counts[i]],
                        initializer = 'glorot_uniform',
                        name = name,
                        trainable = True)
                setattr(self, name, weights)        
        self.built = True
        
    def call(self, inputs):
        '''
        to apply convolution to each word, I reshape the gathered character embeddings to collapse 
        batch and words, i.e. first dimension was previously length batchsize, now is length batchsize*words. I 
        then reshape back after doing convolution. 
            
        For the wtc dataset, encoded has shape (None, 270, 19, 15, 1) before convolution, so need 
        to reshape, as the intention is to apply convolution over each word. Depending on 
        application this might need to be adapted. When adapting, might need to change the return 
        shape of compute_output_shape as well.
        
        Trick taken from the following stackoverflow post: 
        https://stackoverflow.com/questions/51091544/time-distributed-convolutional-layers-in-tensorflow

                
        '''
        
        input_shape = K.shape(inputs)
        inputs = K.cast(inputs, tf.int32)
        encoded = K.gather(self.embeddings, inputs)
        encoded = K.expand_dims(encoded,-1)
        shape = K.int_shape(encoded)
        
        do_reshape = 0
        if do_reshape == 1:
            encoded = K.reshape(encoded,(-1,shape[2],shape[3],shape[4]))
        
        filter_counts = self.filter_counts        
        conv_output = []
        for i in range(len(filter_counts)):
            if filter_counts[i] != 0:
                name = 'kernel' + str(i+1)
                kernel = getattr(self, name)
                c = K.conv2d(encoded, kernel, data_format = 'channels_last')
                c = K.max(c, axis = [1,2])
                conv_output.append(c)
                
        conv_output = K.concatenate(conv_output)
        if do_reshape == 1:
            conv_output = K.reshape(conv_output, (-1, shape[1], self.conv_output_len))
        return conv_output

        
    def compute_output_shape(self,input_shape):
        
        do_reshape = 0
        if do_reshape == 1:
            output_shape = (input_shape[0], input_shape[1], self.conv_output_len)
        else:
            output_shape = (input_shape[0], self.conv_output_len)
        return output_shape
    
    def get_config(self):
        config = {
                'num_vocab': self.num_vocab,
                'embed_dim': self.embed_dim,
                'filter_counts': self.filter_counts
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))     


from keras.layers.recurrent import RNN, GRUCell
import keras.backend as K
from keras.regularizers import l2
class cnn_lstm_layer(RNN):
    """
    The cnn_lstm_layer is similar to the cnn_layer except it adds an lstm after the convolution 
    operation. 
    
    Note that this is not a bidirectional lstm.
    """
    def __init__(self, num_vocab, embed_dim, units=10,filter_counts = None, **kwargs):
        if filter_counts == None:
            raise Exception('must specify filter counts!')     
        
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.units = units
        self.filter_counts = filter_counts
        self.conv_output_len = sum(self.filter_counts)        
        cell = GRUCell(units,kernel_regularizer = l2(0.00), recurrent_regularizer = l2(0.00))
        # super().__init__(cell)
        self.cell = cell
        super().__init__(cell, **kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings')
        
        filter_counts = self.filter_counts        
        for i in range(len(filter_counts)):
            if filter_counts[i] != 0:
                name = 'kernel'+str(i+1)
                weights = self.add_weight(
                        shape = [i+1,self.embed_dim,1,filter_counts[i]],
                        initializer = 'glorot_uniform',
                        name = name)
                setattr(self, name, weights)        

        step_input_shape = [None, self.conv_output_len]
        self.cell.build(step_input_shape)

        self.built = True

    def call(self, inputs):
        inputs = K.cast(inputs, tf.int32)
        encoded = K.gather(self.embeddings, inputs)
        #        encoded = K.print_tensor(encoded, message = 'encoded: ')
        #        encoded = K.tf.Print(encoded, data = [encoded],message='encoded: ',summarize = 1000)
        encoded = K.expand_dims(encoded, -1)
        input_shape = K.int_shape(encoded)
        _, s_words, s_char, s_embed_dim, _ = input_shape
        
        do_reshape = 1
        if do_reshape == 1:
            encoded = K.reshape(encoded, (-1, s_char, s_embed_dim, 1))
        
        filter_counts = self.filter_counts
        conv_output = []
        for i in range(len(filter_counts)):
            if filter_counts[i] != 0:
                name = 'kernel' + str(i+1)
                kernel = getattr(self, name)

                #pad in front
                paddings = [[0, 0], [i, 0], [0, 0], [0, 0]]
                c = K.conv2d(tf.pad(encoded, paddings),
                             kernel, 
                             data_format = 'channels_last',
                             padding = 'valid')
                conv_output.append(c)        
                
        conv_output = K.concatenate(conv_output, axis = 3)
        conv_output = K.squeeze(conv_output,2)
        
        initial_state = self.get_initial_state(conv_output)
        last_output, outputs, states = K.rnn(self.cell.call,
                                                conv_output,
                                                initial_state)
        output = last_output
        
        if do_reshape == 1:
            output = K.reshape(output,(-1,s_words,self.units))
        return output

    def compute_output_shape(self, input_shape):
        batch_size, s_words, s_char = input_shape
        output_shape = (batch_size, s_words,self.units)
        return output_shape


    @property
    def trainable_weights(self):
        return self._trainable_weights + self.cell.trainable_weights
    
    def get_config(self):
        config = {
                'num_vocab': self.num_vocab,
                'embed_dim': self.embed_dim,
                'filter_counts': self.filter_counts,
                'units': self.units
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))      