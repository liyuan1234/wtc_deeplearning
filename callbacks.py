#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:12:46 2018

@author: liyuan
"""

import keras
import keras.backend as K
import numpy as np
import time

class printWeightsEpoch(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs):
        weights = self.model.get_weights()
        weights_flat = np.concatenate([w.flatten() for w in weights])
        mean = np.mean(np.abs(weights_flat))
        for w in weights:
            print(w.flatten()[:5])
        print('  mean weight: {:.7f}'.format(mean), end = '\n')
            
        
        
class printWeightsBatch(keras.callbacks.Callback):
    def on_batch_begin(self,batch,logs):
        weights = self.model.get_weights()
        weights = np.concatenate([w.flatten() for w in weights])
        mean = np.mean(np.abs(weights_flat))
        print('  mean weight: {:.5f}'.format(mean))
        
        
class printLearningRate(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs):
        print('learning rate: {:.5f}'.format(model.optimizer.lr))        
        
      
class accuracyMetricTrainEnd(keras.callbacks.Callback):
    def on_train_end(self,logs):
        self.model.doubledot.predict(subset = 1, verbose = 0)
        
class accuracyMetricEpochEnd(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        self.model.doubledot.predict(subset = 1, verbose = 0)

class printLosses(keras.callbacks.Callback):
    def __init__(self,print_list = 0):
        self.print_list = print_list
        super().__init__()
    
    def on_batch_end(self,batch,logs):
        pass
#        print('\n')
#        self.model.doubledot.printLosses(print_list = self.print_list)
    
    def on_epoch_end(self,epoch,logs):
        self.model.doubledot.printLosses(print_list = self.print_list)
        pass
        
class historyEveryBatch(keras.callbacks.Callback):        
    loss = []
    iteration = 0
    epoch_decimals = []
        
    def on_batch_end(self, batch, logs = {}):
        self.loss.append(logs['loss'])
        
    def on_train_end(self,logs = {}):
        self.iteration= self.iteration + 1
        
class printLearningRate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print('learning_rate: {:.5f}'.format(K.eval(lr_with_decay)))
        
        
def get_rms(weights):
    '''calculates rms of a list of arrays of different shapes
    '''
    weights = np.concatenate([w.flatten() for w in weights])
    mean = np.mean(weights**2)**0.5
    return mean


