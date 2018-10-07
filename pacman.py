#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:28:32 2018

@author: liyuan
"""

import gym
import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import time
import keras
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model


#%% local functions

def smooth(x,n):
    window = np.ones(n)/n
    return np.convolve(x,window,'same')



env = gym.make('MsPacman-v0')
n_actions = env.action_space.n
y = 0.95
lr = 0.8

jList = []
rList = []



#%% define model

s_in = Input(shape = [210,160,3],dtype = 'float32')
conv1 = Conv2D(32,[3,3],padding = 'same',activation = 'relu')(s_in)
conv1 = MaxPooling2D(pool_size = [2,2],strides = 2)(conv1)
f1 = Flatten()(conv1)
f1 = Dense(10)(f1)
Q_predict = Dense(n_actions)(f1)

model = Model(inputs = s_in,outputs = Q_predict)
model.compile(optimizer = keras.optimizers.adam(lr = 0.01),loss = 'mean_squared_error',metrics = [])

print(model.summary())
#%% simulate



#%% simulate


render = 1

for i in range(1000):
    s = env.reset()
    a = env.action_space.sample()
    Q = np.zeros([1,n_actions])
    r_final = 0
    for j in range(10000):
#         if j%10000 == 0:
#             print('\trunning step {}...'.format(j))
        
        s1,r,d,log = env.step(a)
        s1 = s1.astype(np.float32)
        Q1 = model.predict(s1[np.newaxis,:])
        
        Q_target = Q
        Q_target[:,a] = Q_target[:,a]+lr*(r+y*np.max(Q1) - Q_target[:,a])
        
        #s_gray = np.mean(s,axis = 2)
        
        model.fit(x = s[np.newaxis,:],y = Q_target,epochs = 1,verbose = False)
        
        r_final += r
        s = s1
        Q = Q1
        a = np.argmax(Q)
        if np.random.random()>0.9:
            a = env.action_space.sample()
    
        if render == 1:
            env.render()
#            print(r)
            pass
        if d == 1:
            if r == 0:
    #            print('fell into hole!')
                pass
            if r == 1:
    #            print('success!')
    #            print(sess.run(W))
    #            input()
                pass
            
            jList.append(j)
            rList.append(r_final)
            print('running iteration {}, final score is: {}'.format(i,r_final))
            break



#%%

