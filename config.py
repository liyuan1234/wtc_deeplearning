#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:52:25 2018

@author: liyuan
"""

import time
from load_glove_embeddings import load_glove_embeddings
import keras


num_hidden_units = 10
dropout_rate = 0.5
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate)

hyperparameters = {}
hyperparameters['num_hidden_units'] = num_hidden_units
hyperparameters['dropout_rate'] = dropout_rate
hyperparameters['learning_rate'] = learning_rate
hyperparameters['optimizer'] = optimizer


word2index, embedding_matrix = load_glove_embeddings('./embeddings/glove.840B.300d.txt', embedding_dim=300) 
