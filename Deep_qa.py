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
    acc = [0,0,0]
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

        
    def load_model(self, model_creation_function,num_hidden_units = 10):
        training_model,prediction_model,Wsave,title = model_creation_function(self.data,num_hidden_units)
        
        self.training_model = training_model
        self.prediction_model = prediction_model
        self.num_hidden_units = num_hidden_units
        self.Wsave = Wsave
        self.title = title
        
    def train(self,num_iter = 20, learning_rate = 0.001, decay = 1e-6, batch_size = 8, fits_per_iteration = 5,save_plot = 0):
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
            history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = batch_size,epochs = fits_per_iteration)
            self.val_loss = np.append(self.val_loss,history.history['val_loss'])
            self.training_loss = np.append(self.training_loss,history.history['loss'])
        
        save_plot = 0
        self.plot_losses(save_plot)
        
    def adapt_embeddings(self,num_iter = 5,fits_per_iteration = 1,batch_size = 128):
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
                history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = batch_size,epochs = fits_per_iteration)
                history_cache[i] = history.history
                self.val_loss = np.append(self.val_loss,history.history['val_loss'])
                self.training_loss = np.append(self.training_loss,history.history['loss'])
                
        training_model.get_layer('glove_embedding').trainable = False

    def run_many_times(self,num_runs = 5,num_iter = 20, learning_rate = 0.001, decay = 0, batch_size = 128, fits_per_iteration = 5,save_plot = 0, verbose_flag = False):
        training_model = self.training_model
        explain_intseq = self.data.exp_intseq
        questions_intseq = self.data.questions_intseq
        answers_intseq = self.data.answers_intseq
        [train_indices, val_indices, test_indices] = self.data.indices

        dummy_labels_train = self.data.dummy_labels_train
        dummy_labels_val = self.data.dummy_labels_val
        answers_intseq2_val = self.data.answers_intseq2_val
        
        OPTIMIZER = keras.optimizers.Adam(lr = learning_rate,decay = decay)

        
        with tf.device('/cpu:0'):    
            for i in range(num_runs):
                print('running run no. {} of {} runs...'.format(i+1,num_runs))
                training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])        
                self.reset_weights()
                self.reset_losses()
                    
                for j in range(num_iter):
                    try:
                        print('running iteration {}... training/val losses: {:.3f}/{:.3f}'.format(j+1,self.training_loss[-1],self.val_loss[-1]))
                    except IndexError:
                        print('running iteration {}...'.format(j+1))
                    
                    answers_intseq2 = self.data.sample_wrong_answers()
                    X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
                    X_val = [explain_intseq[val_indices],questions_intseq[val_indices],answers_intseq[val_indices],answers_intseq2_val[val_indices]]
                    history = training_model.fit(x = X_train,y = dummy_labels_train,validation_data = [X_val,dummy_labels_val],batch_size = batch_size,epochs = fits_per_iteration, verbose = verbose_flag)
                    self.val_loss = np.append(self.val_loss,history.history['val_loss'])
                    self.training_loss = np.append(self.training_loss,history.history['loss'])
                    
                min_loss = np.min(self.val_loss)
                if min_loss < 1.0:
                    self.make_predictions()
                    self.save_model()
                
                save_plot = 0
                self.plot_losses(save_plot = 0)
            
            self.plot_losses_many_runs(save_plot = 0)

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
        
    def plot_losses(self,losses = None, save_plot = 0):
        if losses == None:
            training_loss = self.training_loss
            val_loss = self.val_loss
        else:
            training_loss,val_loss = losses
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
        loss_cache = self.loss_cache
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
        
    def plot_losses_separately(self, save_plot = 0):
        loss_cache = self.loss_cache
        title = self.title
        
        if title == None:
            title = 'unknown_model'
            
        for i in range(len(loss_cache)):
            losses = loss_cache[i]
            self.plot_losses(losses, save_plot = save_plot)
            plt.show()
        
#%%
        
if __name__ == '__main__':
    temp = Deep_qa()
    temp.load_data()
    temp.load_model(models.cnn)
    #temp.adapt_embeddings()
    temp.run_many_times(self,num_runs = 5,num_iter = 1, learning_rate = 0.001, decay = 0, batch_size = 256, num_epochs = 5,save_plot = 1)