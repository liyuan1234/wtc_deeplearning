#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:59:08 2018

@author: liyuan
"""

from loss_functions import *
from loss_functions import _loss_tensor
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import datetime
import time

import pickle
from Struct import Struct
import _deepqa_misc
import _deepqa_main


class Deep_qa(Struct):
    data = None
    training_model = None
    prediction_model = None
    units = 0
    Wsave = None
    loss_cache = []
    training_loss = np.array([])
    val_loss = np.array([])
    predictions_cache = None
    train_params = Struct()
    model_params = Struct()
    flag = None
    model_flag = None
    acc = None
    
    def __init__(self):
        pass
    
    def load_data(self,flag = 'word'):
        if flag == 'word':
            self.data = Data()
        elif flag == 'char':
            self.data = Data_char()
        else: 
            raise Exception('invalid flag')
        self.data.preprocess_data()
        self.flag = flag
        
#        self.exp_intseq = self.data.exp_intseq
#        self.embedding_matrix = self.data.embedding_matrix
#        self.questions_intseq = self.data.questions_intseq
#        self.answers_intseq = self.data.answers_intseq

        
    def load_model(self, model_creation_function,units = 10,**kwargs):
        training_model,prediction_model,Wsave, model_flag = model_creation_function(self.data,units,**kwargs)
        
        self.training_model = training_model
        self.prediction_model = prediction_model
        self.units = units
        self.Wsave = Wsave
        self.model_flag = model_flag

        trainable_count, untrainable_count = self.count_params()
        self.model_params.units_char = None
        self.set_params(header = 'model_params', 
                        units = units,
                        trainable_count = trainable_count,
                        untrainable_count = untrainable_count,
                        rnn_type = 'GRU',
                        cutoff_length = 150,
                        **kwargs)
        self.set_params(header = 'train_params', adapt_embeddings = 0)
        

        
        
        
    def train(self,num_iter = 20, learning_rate = 0.001, decay = 0, batch_size = 16, fits_per_iteration = 5, save_plot = 0,verbose = 1, callbacks = None):
        
        self.set_params(header = 'train_params',
                        num_iter = num_iter,
                        lr = learning_rate,
                        decay = decay,
                        batch_size = batch_size,
                        fits_per_iteration = fits_per_iteration,
                        optimizer = 'adam')
        
        if callbacks == None:
            callbacks = []
        
        training_model = self.training_model
        prediction_model = self.prediction_model
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers_intseq = self.data.answers_intseq
        [train_indices, val_indices, test_indices] = self.data.indices

        dummy_labels_train = self.data.dummy_labels_train
        dummy_labels_val = self.data.dummy_labels_val
        answers_intseq2_val = self.data.answers_intseq2_val 
        
        training_model.doubledot = self
        
        OPTIMIZER = keras.optimizers.Adam(lr = learning_rate,decay = decay)
        training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
        
        

        for i in range(num_iter):
            start_time = time.time()
            print('running iteration {}...'.format(i+1), end = '')
                
            answers_intseq2 = self.data.sample_wrong_answers()
            X_train = [explain_intseq[train_indices],
                       questions_intseq[train_indices],
                       answers_intseq[train_indices],
                       answers_intseq2[train_indices]]
            X_val = [explain_intseq[val_indices],
                     questions_intseq[val_indices],
                     answers_intseq[val_indices],
                     answers_intseq2_val[val_indices]]
            history = training_model.fit(x = X_train,
                                         y = dummy_labels_train,
                                         validation_data = [X_val,dummy_labels_val],
                                         batch_size = batch_size,
                                         epochs = fits_per_iteration,
                                         verbose = verbose,
                                         callbacks = callbacks)
            self.val_loss = np.append(self.val_loss,history.history['val_loss'])
            self.training_loss = np.append(self.training_loss,history.history['loss'])
            
            print('training/val losses: {:.3f}/{:.3f} ... time taken is {:.2f}s'.format(self.training_loss[-1],self.val_loss[-1], time.time() - start_time))
        self.plot_losses(save_plot = save_plot)
        
        
    def set_params(self, header = None, **kwargs):
        '''
        sets parameters for self.header e.g. self.train_params or self.model_params
        '''
        if header == None:
            raise('must specify header field!!')
        if not hasattr(self,header):
            raise('header field does not exist!')
        for arg in kwargs:
            setattr(getattr(self,header),arg,kwargs[arg])

        
    def predict(self, subset = 1, verbose = 1):
        cache = _deepqa_main.predict(self,subset, verbose)
        return cache
        
        
    def adapt_embeddings(self,lr = 0.001, num_iter = 5,fits_per_iteration = 1,batch_size = 16, embeddings_verbose_flag = 1):
        _deepqa_main.adapt_embeddings(self,
                                      lr = lr,
                                      num_iter = num_iter, 
                                      fits_per_iteration = fits_per_iteration,
                                      batch_size = batch_size,
                                      embeddings_verbose_flag = embeddings_verbose_flag)

    def run_many_times(self,num_runs = 5,num_iter = 20, learning_rate = 0.001, decay = 0, batch_size = 64, fits_per_iteration = 5,save_plot = 0, verbose = False, embeddings_verbose_flag = False, adapt_embeddings = False, adapt_iteration = 5):
        _deepqa_main.run_many_times(self,
                                num_runs,
                                num_iter,
                                learning_rate,
                                decay,
                                batch_size,
                                fits_per_iteration,
                                save_plot,verbose,
                                embeddings_verbose_flag,
                                adapt_embeddings,
                                adapt_iteration)

    def reset_weights(self):
        self.training_model.set_weights(self.Wsave)
        self.acc = [0,0,0]
    
    def reset_losses(self):
        self.loss_cache.append([self.training_loss,self.val_loss])
        self.training_loss = np.array([])
        self.val_loss = np.array([])
        
    def clear_losses(self):
        self.training_loss = np.array([])
        self.val_loss = np.array([])
        
    def summary(self):
        print(self.training_model.summary())
        print(self.model_params)
        print(self.train_params)
        
    def save_obj(self):
        _deepqa_misc.save_obj(self)
        
    def save_model(self):
        _deepqa_misc.save_model(self)
        
    def plot_losses(self,losses = None, save_plot = 0, maximize_yaxis = 1):
        _deepqa_misc.plot_losses(self, losses, save_plot, maximize_yaxis)
        
    def plot_losses_many_runs(self, save_plot = 0):
        _deepqa_misc.plot_losses_many_runs(self, save_plot)
        
    def plot_losses_separately(self, save_plot = 0):
        _deepqa_misc.plot_losses_separately(self, save_plot)
            
    def save_losses(self,title = None):
        _deepqa_misc.save_losses(self, title)

    def calculate_loss(self, flag = 'val',print_list = 0):
        _deepqa_misc.calculate_loss(self,flag,print_list)

    def printLosses(self, print_list = 0):
        _deepqa_misc.printLosses(self,print_list)
        
    def count_params(self):
        trainable_count, untrainable_count = _deepqa_misc.count_params(self)
        return trainable_count, untrainable_count
    def get_formatted_title(self):
        title = _deepqa_misc.get_formatted_title(self)
        return title
    
    def load_obj(file):
        obj = _deepqa_misc.load_obj(file)
        return obj
        
#%% example script
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        

from Data import Data
from Data_char import Data_char
import models
import models_char
from callbacks import printWeightsEpoch, printWeightsBatch, printLosses, historyEveryBatch, printLearningRate, accuracyMetricEpochEnd, accuracyMetricTrainEnd


if __name__ == '__main__':
    temp = Deep_qa()
    embedding_flag = 'char'
    if embedding_flag == 'char':
        temp.load_data('char')
        temp.load_model(models_char.model,
                        units = 10, 
                        units_char = 10,
                        threshold = 0.5,
                        model_flag = 'cnn_lstm',
                        filter_counts = [10,10,10,10,10,10])
    elif embedding_flag == 'word':
        temp.load_data('word')
        temp.load_model(models.model,
                    units = 20,
                    model_flag = 'normal',
                    filter_nums = None,
                    rnn_layers = 3,
                    threshold = 0.5,
                    reg = 0.00,
                    dropout_rate = 0.5,)
    temp.summary()
#    temp.adapt_embeddings()
    historyEveryBatchCallback = historyEveryBatch()
    printLossesCallback = printLosses(print_list = 0)
    
    temp.train(num_iter = 50,
               fits_per_iteration = 5,
               verbose = 1,
               batch_size = 64,
               learning_rate = 0.01,
               decay = 1e-3,
               callbacks = [printLearningRate(),
                            printLossesCallback,
                            historyEveryBatchCallback,
                            accuracyMetricEpochEnd()])
    plt.plot(historyEveryBatchCallback.epoch_decimals,historyEveryBatchCallback.loss)
    plt.ylim([0,1.5])
    
    
    cache = temp.predict()
