#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:12:46 2018

@author: liyuan
"""

import keras
import numpy as np

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
        print('  mean weight: {:.7f}'.format(mean))
        
        
class printLearningRate(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs):
        print('learning rate: {:.5f}'.format(model.optimizer.lr))        
        
        
def get_rms(weights):
    '''calculates rms of a list of arrays of different shapes
    '''
    weights = np.concatenate([w.flatten() for w in weights])
    mean = np.mean(weights**2)**0.5
    return mean
