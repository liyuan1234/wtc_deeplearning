#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:59:08 2018

@author: liyuan
"""
from Data import Data
from loss_functions import *
from loss_functions import _loss_tensor
import models

import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import datetime


class Deep_qa:
    data = None
    training_model = None
    prediction_model = None
    num_hidden_units = 0
    Wsave = None
    loss_cache = []
    training_loss = np.array([])
    val_loss = np.array([])
    acc = []
    title = None
    
    def __init__(self):
        pass
    
    def load_data(self):
        self.data = Data()
        self.data.preprocess_data()
        
#        self.exp_intseq = self.data.exp_intseq
#        self.embedding_matrix = self.data.embedding_matrix
#        self.questions_intseq = self.data.questions_intseq
#        self.answers_intseq = self.data.answers_intseq

        
    def load_model(self, function_handle,num_hidden_units = 10):
        training_model,prediction_model,Wsave,title = function_handle(self.data,num_hidden_units)
        
        self.training_model = training_model
        self.prediction_model = prediction_model
        self.num_hidden_units = num_hidden_units
        self.Wsave = Wsave
        self.title = title
        
    def train(self,num_iter = 20, learning_rate = 0.001, decay = 1e-6, batch_size = 8, num_epochs = 5,save_plot = 0):
        training_model = self.training_model
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers_intseq = self.data.answers_intseq
        [train_indices, val_indices, test_indices] = self.data.indices

        dummy_labels_train = self.data.dummy_labels_train
        dummy_labels_val = self.data.dummy_labels_val
        answers_intseq2_val = self.data.answers_intseq2_val 
#        print(training_model is self.training_model) # True
        
        OPTIMIZER = keras.optimizers.Adam(lr = learning_rate,decay = decay)
        training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
        

        for i in range(num_iter):
            print('running iteration {}...'.format(i+1))
            answers_intseq2 = self.data.sample_wrong_answers()
            X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
            X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices], answers_intseq2_val[val_indices]]
            history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = batch_size,epochs = num_epochs)
            self.val_loss = np.append(self.val_loss,history.history['val_loss'])
            self.training_loss = np.append(self.training_loss,history.history['loss'])
        
        save_plot = 0
        self.plot_loss_history(save_plot)
        
    def adapt_embeddings(self,num_iter = 5,num_epochs = 1,batch_size = 128):
        training_model = self.training_model
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers_intseq = self.data.answers_intseq
        [train_indices, val_indices, test_indices] = self.data.indices

        dummy_labels_train = self.data.dummy_labels_train
        dummy_labels_val = self.data.dummy_labels_val
        answers_intseq2_val = self.data.answers_intseq2_val 
        
        training_model.get_layer('glove_embedding').trainable = True
        training_model.compile(optimizer = keras.optimizers.Adam(0.0003),loss = _loss_tensor,metrics = [])
        
        history_cache = dict()
        
        with tf.device('/cpu:0'):
            for i in range(num_iter):
                answers_intseq2 = self.data.sample_wrong_answers()
                X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
                X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices],answers_intseq2_val[val_indices]]
                history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = batch_size,epochs = num_epochs)
                history_cache[i] = history.history
                self.val_loss = np.append(self.val_loss,history.history['val_loss'])
                self.training_loss = np.append(self.training_loss,history.history['loss'])
                
        training_model.get_layer('glove_embedding').trainable = False

    def run_many_times(self,num_runs = 5,num_iter = 20, learning_rate = 0.001, decay = 0, batch_size = 128, num_epochs = 5,save_plot = 0):
        training_model = self.training_model
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers_intseq = self.data.answers_intseq
        [train_indices, val_indices, test_indices] = self.data.indices

        dummy_labels_train = self.data.dummy_labels_train
        dummy_labels_val = self.data.dummy_labels_val
        answers_intseq2_val = self.data.answers_intseq2_val
        
        
        with tf.device('/cpu:0'):    
            for run in range(num_runs):
                training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])        
                training_model.set_weights(Wsave)
                self.reset_losses()
                    
                for i in range(num_iter):
                    print('running iteration {}...'.format(i+1))
                    answers_intseq2 = self.data.sample_wrong_answers()
                    X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
                    X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices],answers_intseq2_val[val_indices]]
                    history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = 8,epochs = 5)
                    self.val_loss = np.append(val_loss,history.history['val_loss'])
                    self.training_loss = np.append(training_loss,history.history['loss'])
                
                save_plot = 0
                self.plot_loss_history(save_plot = 0)
                self.reset_losses()
            
            self.plot_losses_many_runs(loss_cache,'cnn_model_10units')


    def reset_losses(self):
        self.loss_cache.append([self.training_loss,self.val_loss])
        self.training_loss = np.array([])
        self.val_loss = np.array([])
        
    def make_predictions(self):
        prediction_model = self.prediction_model
        all_answer_options_intseq = self.data.all_answer_options_intseq
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers = self.data.answers
        
        prediction_model.compile(optimizer = 'adam', loss = lambda y_true,y_pred: y_pred, metrics = [keras.metrics.categorical_accuracy])
        
        
        all_answer_options_intseq = np.array(all_answer_options_intseq)
        
        input1 = explain_intseq
        input2 = questions_intseq
        input3 = all_answer_options_intseq[:,0,:]
        input4 = all_answer_options_intseq[:,1,:]
        input5 = all_answer_options_intseq[:,2,:]
        input6 = all_answer_options_intseq[:,3,:]  
            
        predict_output = prediction_model.predict([input1,input2,input3,input4,input5,input6],batch_size = 1)
        predicted_ans = np.argmax(predict_output,axis = 1)
        print(predict_output)
        print(predicted_ans)
        
        
        int_ans = np.array([convert_to_int(letter) for letter,ans in answers])
        train_acc = np.mean(predicted_ans[train_indices] == int_ans[train_indices])
        val_acc = np.mean(predicted_ans[val_indices] == int_ans[val_indices])
        test_acc = np.mean(predicted_ans[test_indices] == int_ans[test_indices])
        
        print('train,val,test accuracies: {:.2f}/{:.2f}/{:.2f}'.format(train_acc,val_acc,test_acc))
        
        self.acc = [train_acc,val_acc,test_acc]
    
    def save_model(self):
        train_acc,val_acc,test_acc = self.acc
        val_loss = self.val_loss
        num_hidden_units = self.num_hidden_units
        training_model = self.training_model
        title = self.title
        
        stats = '_{}units_{:.3f}_{:.2f}_{:.2f}_{:.2f}'.format(num_hidden_units,np.min(val_loss),train_acc,val_acc,test_acc)
        filepath = './saved_models/' + title + stats + '.h5'
        training_model.save_weights(filepath)
        
    def plot_loss_history(self, save_plot = 0):
        training_loss = self.training_loss
        val_loss = self.val_loss
        title = self.title
        
        plt.plot(val_loss, label = 'validation loss')
        plt.plot(training_loss, label = 'training loss')
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch num')
        plt.title(title)
        if save_plot == 1:
    #        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
            loss_str = '_{:.3f}'.format(np.min(val_loss))
            plt.savefig('./images/loss_'+title+'_'+loss_str+'.png')
        plt.show()
        
    def plot_losses_many_runs(self, save_plot = 0):
        
        title = self.title
        if title == None:
            title = 'unknown_model'
            
        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
        filepath = './images/loss_'+title+timestamp
        
        for training_losses,validation_losses in loss_cache:
            plt.plot(training_losses,'r')
            plt.plot(validation_losses,'b')
        plt.plot(training_losses,'r',label = 'training loss')
        plt.plot(validation_losses,'b',label = 'validation loss')  
        plt.legend()
        if save_plot == 1:
            num_runs = len(loss_cache)
            plt.savefig(filepath + '_{}runs'.format(num_runs))
            print(title)
        plt.show()