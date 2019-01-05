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
    def __init__(self, num_vocab, embed_dim, fcounts = None):
        if fcounts == None:
            raise('must specify filter counts!')
        
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.fcounts = fcounts
        super().__init__()
        
    def build(self, input_shape):
        f1_count,f2_count,f3_count,f4_count,f5_count,f6_count = self.fcounts
        
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings',
                trainable = True)
        self.kernel1 = self.add_weight(
                shape = [1,self.embed_dim,1,f1_count],
                initializer = 'glorot_uniform',
                name = 'f1',
                trainable = True)          
        self.kernel2 = self.add_weight(
                shape = [2,self.embed_dim,1,f2_count],
                initializer = 'glorot_uniform',
                name = 'f2',
                trainable = True)
        self.kernel3 = self.add_weight(
                shape = [3,self.embed_dim,1,f3_count],
                initializer = 'glorot_uniform',
                name = 'f3',
                trainable = True)
        self.kernel4 = self.add_weight(
                shape = [4,self.embed_dim,1,f4_count],
                initializer = 'glorot_uniform',
                name = 'f4',
                trainable = True)
        self.kernel5 = self.add_weight(
                shape = [5,self.embed_dim,1,f5_count],
                initializer = 'glorot_uniform',
                name = 'f5',
                trainable = True)  
        self.kernel6 = self.add_weight(
                shape = [6,self.embed_dim,1,f6_count],
                initializer = 'glorot_uniform',
                name = 'f6',
                trainable = True)          
        self.conv_output_len = sum(self.fcounts)
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
        c1 = K.conv2d(encoded,self.kernel1,data_format = 'channels_last')
        c1 = K.max(c1, axis = [1,2])        
        c2 = K.conv2d(encoded,self.kernel2,data_format = 'channels_last')
        c2 = K.max(c2, axis = [1,2])
        c3 = K.conv2d(encoded,self.kernel3,data_format = 'channels_last')
        c3 = K.max(c3, axis = [1,2])
        c4 = K.conv2d(encoded,self.kernel4,data_format = 'channels_last')
        c4 = K.max(c4, axis = [1,2])
        c5 = K.conv2d(encoded,self.kernel5,data_format = 'channels_last')
        c5 = K.max(c5, axis = [1,2])
        c6 = K.conv2d(encoded,self.kernel6,data_format = 'channels_last')
        c6 = K.max(c6, axis = [1,2])        
        c = K.concatenate([c1,c2,c3,c4,c5,c6])
        conv_encoded = K.reshape(c,(-1,shape[1],self.conv_output_len))
        
        return conv_encoded
        
    def compute_output_shape(self,input_shape):
        output_shape = (input_shape[0],input_shape[1],self.conv_output_len)
        return output_shape

from keras.layers.recurrent import RNN, GRUCell
import keras.backend as K
from keras.regularizers import l2
class cnn_lstm_layer(RNN):
    def __init__(self, num_vocab, embed_dim, units=10,fcounts = None):
        if fcounts == None:
            raise('must specify filter counts!')    
        
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.units = units
        self.fcounts = fcounts
        cell = GRUCell(units,kernel_regularizer = l2(0.00), recurrent_regularizer = l2(0.00))
        # super().__init__(cell)
        self.cell = cell
        super().__init__(cell)

    def build(self, input_shape):
        f1_count,f2_count,f3_count,f4_count,f5_count,f6_count = self.fcounts

        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim],
                initializer = 'glorot_uniform',
                name = 'char_embeddings')
        self.kernel1 = self.add_weight(
                shape = [1,self.embed_dim,1,f1_count],
                initializer = 'glorot_uniform',
                name = 'f1')          
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
        self.kernel5 = self.add_weight(
                shape = [5,self.embed_dim,1,f5_count],
                initializer = 'glorot_uniform',
                name = 'f5')  
        self.kernel6 = self.add_weight(
                shape = [6,self.embed_dim,1,f6_count],
                initializer = 'glorot_uniform',
                name = 'f6')          
        self.conv_output_len = sum(self.fcounts)
        step_input_shape = [None, self.conv_output_len]
        self.cell.build(step_input_shape)

        self.built = True

    def call(self, inputs):
        '''
        uses the following reshaping trick
        https://stackoverflow.com/questions/51091544/time-distributed-convolutional-layers-in-tensorflow
        '''
        inputs = K.cast(inputs, tf.int32)
        encoded = K.gather(self.embeddings, inputs)
        #        encoded = K.print_tensor(encoded, message = 'encoded: ')
        #        encoded = K.tf.Print(encoded, data = [encoded],message='encoded: ',summarize = 1000)
        encoded = K.expand_dims(encoded, -1)
        input_shape = K.int_shape(encoded)
        _, s_words, s_char, s_embed_dim, _ = input_shape
        encoded = K.reshape(encoded, (-1, s_char, s_embed_dim, 1))
        paddings1 = [[0, 0], [0, 0], [0, 0], [0, 0]]
        paddings2 = [[0, 0], [1, 0], [0, 0], [0, 0]]
        paddings3 = [[0, 0], [2, 0], [0, 0], [0, 0]]
        paddings4 = [[0, 0], [3, 0], [0, 0], [0, 0]]
        paddings5 = [[0, 0], [4, 0], [0, 0], [0, 0]]
        paddings6 = [[0, 0], [5, 0], [0, 0], [0, 0]]
        
        
        c1 = K.conv2d(tf.pad(encoded, paddings1),
                      self.kernel1,
                      data_format='channels_last',
                      padding='valid')        
        c2 = K.conv2d(tf.pad(encoded, paddings2),
                      self.kernel2,
                      data_format='channels_last',
                      padding='valid')  # shape = (?,19,1,3)
        c3 = K.conv2d(tf.pad(encoded, paddings3),
                      self.kernel3,
                      data_format='channels_last',
                      padding='valid')
        c4 = K.conv2d(tf.pad(encoded, paddings4),
                      self.kernel4,
                      data_format='channels_last',
                      padding='valid')
        c5 = K.conv2d(tf.pad(encoded, paddings5),
                      self.kernel5,
                      data_format='channels_last',
                      padding='valid')
        c6 = K.conv2d(tf.pad(encoded, paddings6),
                      self.kernel6,
                      data_format='channels_last',
                      padding='valid')        
        
        c = K.concatenate([c1,c2,c3,c4,c5,c6], axis=3)  # shape = (?,19,1,12)
        c = K.squeeze(c, 2)  # shape = (?,19,12)


        initial_state = self.get_initial_state(c)
        last_output, outputs, states = K.rnn(self.cell.call,
                                                c,
                                                initial_state)
        output = last_output
        output = K.reshape(output,(-1,s_words,self.units))
        return output

    def compute_output_shape(self, input_shape):
        batch_size, s_words, s_char = input_shape
        output_shape = (batch_size, s_words,self.units)
        return output_shape


    @property
    def trainable_weights(self):
        return self._trainable_weights + self.cell.trainable_weights
    
    
    
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