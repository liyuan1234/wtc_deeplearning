#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:42:29 2018

@author: liyuan
"""

from loss_functions import cosine_similarity
import tensorflow as tf
import keras.backend as K
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


a1 = np.array([[0,0,1]],dtype = 'float')
a2 = np.array([[0,1,0]],dtype = 'float')
a3 = np.array([[0,0,-1]],dtype = 'float')
a4 = np.array([[-1,1.5,3.5]],dtype = 'float')
a5 = np.array([[-1,-2,-3]],dtype = 'float')

#v1 = tf.constant(a1)
#v2 = tf.constant(a2)
#v3 = tf.constant(a3)
#v4 = tf.constant(a4)
#v5 = tf.constant(a5)
#
#
#cos1 = K.eval(get_cosine_similarity([v1,v1]))
#cos2 = K.eval(get_cosine_similarity([v1,v3]))
#cos3 = K.eval(get_cosine_similarity([v1,v2]))
#cos4 = K.eval(get_cosine_similarity([v4,v5]))
#
#print('expected cos1: 1.0 predicted cos1: {:.1f}'.format(cos1[0,0]))
#print('expected cos2: -1.0 predicted cos1: {:.1f}'.format(cos2[0,0]))
#print('expected cos3: 0.0 predicted cos1: {:.1f}'.format(cos3[0,0]))
#print('expected cos4: -0.848556 predicted cos1: {:.6f}'.format(cos4[0,0]))


#%%

#a = np.vstack([a1,a1,a1,a4])
#b = np.vstack([a1,a3,a2,a5])
#
#print(a)
#print(b)
#
#v1 = tf.constant(a)
#v2 = tf.constant(b)
#
#cos = K.eval(get_cosine_similarity([v1,v2]))
#
#print(cos)



#%%

import tensorflow as tf
import numpy as np

v1 = np.array([[0,0,1],[0,0,1],[0,0,1]],dtype = 'float')
v2 = np.array([[0,0,1],[0,1,0],[0,0,-1]],dtype = 'float')

s = cosine_similarity([v1,v2])
print(tf.Session().run([s]))


#%%

from loss_functions import hinge_loss

'''
need to check that hinge loss returns a vector of loss for each element. Should not be returning the mean or summed loss

'''




a1 = np.array([1,2,3,4],dtype = 'float32').reshape([4,1])
a2 = np.array([0.8,1.8,2.8,3.5],dtype = 'float32').reshape([4,1])

v1 = tf.constant(a1)
v2 = tf.constant(a2)

print(K.eval(hinge_loss([v1,v2])))