#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:35:28 2018

@author: liyuan
"""

import keras
from keras.engine.topology import Layer
from keras import layers
from keras import Model
import keras.backend as K
import tensorflow as tf

class cnn_layer(Layer):
    def __init__(self, num_vocab, embed_dim):
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        
        super().__init__()
        
    def build(self, input_shape):
        f2_count = 16
        f3_count = 16
        f4_count = 16
        
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings')
        self.kernel2 = self.add_weight(
                shape = [2,self.embed_dim,1,f2_count],
                initializer = 'glorot_uniform',
                name = 'f2')
        self.kernel3 = self.add_weight(
                shape = [3,self.embed_dim,1,f3_count],
                initializer = 'glorot_uniform',
                name = 'f3')
        self.kernel4 = self.add_weight(
                shape = [4,self.embed_dim,1,f4_count],
                initializer = 'glorot_uniform',
                name = 'f4')        
        self.conv_output_len = f2_count+f3_count+f4_count
        self.built = True
        
    def call(self, inputs):
        '''
        uses the following trick
        https://stackoverflow.com/questions/51091544/time-distributed-convolutional-layers-in-tensorflow
        '''
        
        input_shape = K.shape(inputs)
        inputs = K.cast(inputs, tf.int32)
        encoded = K.gather(self.embeddings, inputs)
        encoded = K.expand_dims(encoded,-1)
        shape = K.int_shape(encoded)
        encoded = K.reshape(encoded,(-1,shape[2],shape[3],shape[4]))
        c1 = K.conv2d(encoded,self.kernel2,data_format = 'channels_last')
        c1 = K.max(c1, axis = [1,2])
        c2 = K.conv2d(encoded,self.kernel3,data_format = 'channels_last')
        c2 = K.max(c2, axis = [1,2])
        c3 = K.conv2d(encoded,self.kernel4,data_format = 'channels_last')
        c3 = K.max(c3, axis = [1,2])
        c = K.concatenate([c1,c2,c3])
        conv_encoded = K.reshape(c,(-1,shape[1],self.conv_output_len))
        
        return conv_encoded
        
    def compute_output_shape(self,input_shape):
        output_shape = (input_shape[0],input_shape[1],self.conv_output_len)
        return output_shape
    
    
    
    
    

class cnn_lstm_layer(Layer):
    def __init__(self, num_vocab, embed_dim, num_hidden_units = 10):
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.num_hidden_units = num_hidden_units
        super().__init__()
        
    def build(self, input_shape):
        f2_count = 3
        f3_count = 4
        f4_count = 5
        
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings')
        self.kernel2 = self.add_weight(
                shape = [2,self.embed_dim,1,f2_count],
                initializer = 'glorot_uniform',
                name = 'f2')
        self.kernel3 = self.add_weight(
                shape = [3,self.embed_dim,1,f3_count],
                initializer = 'glorot_uniform',
                name = 'f3')
        self.kernel4 = self.add_weight(
                shape = [4,self.embed_dim,1,f4_count],
                initializer = 'glorot_uniform',
                name = 'f4')        
        self.conv_output_len = f2_count+f3_count+f4_count
        
        
        with tf.variable_scope("chars", reuse = tf.AUTO_REUSE):
            self.cell_fw = tf.contrib.rnn.LSTMCell(self.num_hidden_units)
            self.cell_bw = tf.contrib.rnn.LSTMCell(self.num_hidden_units)
        
        
        self.built = True
        
    def call(self, inputs):
        '''
        uses the following reshaping trick
        https://stackoverflow.com/questions/51091544/time-distributed-convolutional-layers-in-tensorflow
        '''
        
        inputs = K.cast(inputs, tf.int32)
        encoded = K.gather(self.embeddings, inputs)
        encoded = K.expand_dims(encoded,-1)
        input_shape = K.int_shape(encoded)
        _, s_words, s_char, s_embed_dim,num_channels = input_shape
        encoded = K.reshape(encoded,(-1,s_char,s_embed_dim,1))
        paddings2 = [[0,0],[1,0],[0,0],[0,0]]
        paddings3 = [[0,0],[2,0],[0,0],[0,0]]
        paddings4 = [[0,0],[3,0],[0,0],[0,0]]
        c2 = K.conv2d(tf.pad(encoded,paddings2),
                      self.kernel2,
                      data_format = 'channels_last',
                      padding = 'valid') # shape = (?,19,1,3)
        c3 = K.conv2d(tf.pad(encoded,paddings3),
                      self.kernel3,
                      data_format = 'channels_last',
                      padding = 'valid')
        c4 = K.conv2d(tf.pad(encoded,paddings4),
                      self.kernel4,
                      data_format = 'channels_last',
                      padding = 'valid')
        c = K.concatenate([c2,c3,c4],axis = 3) #shape = (?,19,1,12)
        c = K.reshape(c,(-1,s_char,self.conv_output_len)) # shape = (?,19,12)
#        conv_encoded = K.reshape(c,(-1,input_shape[1],self.conv_output_len))
#        conv_encoded = tf.unstack(conv_encoded, axis = 1)
        
        # LSTM
#        with tf.variable_scope("chars", reuse = tf.AUTO_REUSE):
#            cell_fw = tf.contrib.rnn.LSTMCell(self.num_hidden_units)
#            H,C = tf.contrib.rnn.static_rnn(cell_fw, conv_encoded, dtype = tf.float32)
#            h = C[1]
                
        
        
            
#            with tf.variable_scope("chars"):
#			# get char embeddings matrix
#			w_fw = tf.get_variable(
#					name="LSTMfw",
#					dtype=tf.float32,
#					shape=[])
#            w_bw = tf.get_variable(
#					name="LSTMbw",
#					dtype=tf.float32,
#					shape=[])
            
            

        _output = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, c, dtype=tf.float32)
		# read and concat final output
        _, ((_, output_fw), (_, output_bw)) = _output
        output = tf.concat([output_fw, output_bw], axis=-1) # shape = (?, 20)
        output = K.reshape(output,(-1,s_words,2*self.num_hidden_units)) # shape = (?, sentence length {270}, 20)
        return output
        
    def compute_output_shape(self,input_shape):
        batch_size, s_words, s_char = input_shape
        output_shape = (batch_size,s_words,2*self.num_hidden_units)
        return output_shape
    
    
    
    
#    def call(self, inputs):
#        input_shape = K.shape(inputs)
#        inputs = K.cast(inputs, tf.int32)
#        encoded = K.gather(self.embeddings, inputs)
#        encoded = K.expand_dims(encoded,-1)
#        split_encoded = K.tf.unstack(encoded, axis = 1)
#        
#        
#        conv_rep = []
#        for e in split_encoded:
#            c1 = K.conv2d(e,self.kernel2,data_format = 'channels_last')
#            c1 = K.max(c1, axis = [1,2])
#            c2 = K.conv2d(e,self.kernel3,data_format = 'channels_last')
#            c2 = K.max(c2, axis = [1,2])
#            c3 = K.conv2d(e,self.kernel4,data_format = 'channels_last')
#            c3 = K.max(c3, axis = [1,2])            
#            c = K.concatenate([c1,c2,c3])
#            conv_rep.append(c)
#
#        conv_encoded = K.stack(conv_rep, axis = 1)
#        return conv_encoded