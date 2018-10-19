#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:53:32 2018

@author: liyuan
"""
import keras.backend as K


def hinge_loss(inputs,hinge_loss_parameter = 1.5):
    similarity1,similarity2 = inputs
#    print(similarity1,similarity2)
    hinge_loss = similarity1 - similarity2 - hinge_loss_parameter
    hinge_loss = -hinge_loss
    loss = K.maximum(0.0,hinge_loss)
    return loss

def _loss_tensor(y_true,y_pred):
    return y_pred

def get_cosine_similarity(input_tensors):
    x,y = input_tensors
    similarity = K.sum(x*y)/get_norm(x)/get_norm(y)
    similarity = K.reshape(similarity,[1,1])
    return similarity

def get_norm(x):
    norm = K.sum(x**2)**0.5
    return norm

# bad, need to do unit normalization
tf_cos_similarity = lambda x: K.tf.reshape(1-K.tf.losses.cosine_distance(x[0],x[1],axis = 1), shape = [1,1])