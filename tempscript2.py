#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:07:21 2018

@author: liyuan
"""


import keras
from keras.engine.topology import Layer
from keras import layers
from keras import Model
import keras.backend as K
import tensorflow as tf

class Char_Embedding(Layer):
    def __init__(self, num_vocab, embed_dim):
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        
        super().__init__()
        
    def build(self, input_shape):
        self.embeddings = self.add_weight(
                shape = [self.num_vocab, self.embed_dim]
                initializer = 'uniform')
        
        
        pass
    
    def call(self, inputs):
        encoded = K.gather(self.embeddings, inputs)
        
        

