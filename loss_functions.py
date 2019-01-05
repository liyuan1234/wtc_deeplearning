#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:53:32 2018

@author: liyuan
"""
import keras.backend as K
import tensorflow as tf

def hinge_loss(inputs,threshold = 1.0):
    similarity1,similarity2 = inputs
    hinge_loss = similarity1 - similarity2 - threshold
    hinge_loss = -hinge_loss
    loss = K.maximum(0.0,hinge_loss)
    return loss

def _loss_tensor(y_true,y_pred):
    return y_pred

'''
got this wrong: 

def get_cosine_similarity(input_tensors):
    x,y = input_tensors
    similarity = K.sum(x*y)/get_norm(x)/get_norm(y)
    similarity = K.reshape(similarity,[1,1])
    return similarity
    
    
    
    
then still have error: 
(get_norm should return a vector of norms, one for each example, 
but instead averages over all the examples in the batch)
    
def get_cosine_similarity(input_tensors):
    x,y = input_tensors
    similarity = K.sum(x*y,axis = 1,keepdims = True)
#    similarity = tf.Print(similarity, [similarity], message = '', summarize = 10)
    norm1 = get_norm(x)
    norm1 = tf.Print(norm1,[norm1], message = '')
    similarity = similarity/norm1/get_norm(y)
    
#    similarity = K.sum(x*y,axis = 1,keepdims = True)/get_norm(x)/get_norm(y)
    return similarity

def get_norm(x):
    norm = K.sqrt(K.sum(K.square(x)))
    return norm
'''

def cosine_similarity(inputs):
#    import tensorflow as tf
    x,y = inputs
    x = K.tf.nn.l2_normalize(x,axis = 1)
    y = K.tf.nn.l2_normalize(y,axis = 1)
    
    s = 1 - tf.losses.cosine_distance(x,y,axis = 1, reduction = tf.losses.Reduction.NONE)
    return s


