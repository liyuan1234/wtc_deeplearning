#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:30:32 2018

@author: liyuan
"""
import keras
import numpy as np
from Struct import Struct


def adapt_embeddings(self,num_iter = 5,fits_per_iteration = 1,batch_size = 16, embeddings_verbose_flag = False):
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
                                         verbose = embeddings_verbose_flag)
            history_cache[i] = history.history
            self.val_loss = np.append(self.val_loss,history.history['val_loss'])
            self.training_loss = np.append(self.training_loss,history.history['loss'])
            
    training_model.get_layer('glove_embedding').trainable = False
    training_model.compile(optimizer = keras.optimizers.Adam(0.001),loss = _loss_tensor,metrics = [])
    self.history_cache = history_cache

def run_many_times(self,num_runs = 5,num_iter = 20, learning_rate = 0.001, decay = 0, batch_size = 128, fits_per_iteration = 5,save_plot = 0, verbose = False, embeddings_verbose_flag = False, adapt_embeddings = False, adapt_iteration = 5):
    training_model = self.training_model
    explain_intseq = self.data.exp_intseq
    questions_intseq = self.data.questions_intseq
    answers_intseq = self.data.answers_intseq
    [train_indices, val_indices, test_indices] = self.data.indices

    dummy_labels_train = self.data.dummy_labels_train
    dummy_labels_val = self.data.dummy_labels_val
    answers_intseq2_val = self.data.answers_intseq2_val
    
    OPTIMIZER = keras.optimizers.Adam(lr = learning_rate,decay = decay)

    for i in range(num_runs):
        print('running run no. {} of {} runs...'.format(i+1,num_runs))       
        self.reset_weights()
        self.reset_losses()
        
        if adapt_embeddings is True:
            self.adapt_embeddings(num_iter = adapt_iteration,
                                  embeddings_verbose_flag = embeddings_verbose_flag)
            
        self.train(num_iter = num_iter,
                   learning_rate = learning_rate,
                   decay = decay,
                   batch_size = batch_size,
                   fits_per_iteration = fits_per_iteration,
                   verbose = verbose,
                   save_plot = save_plot)
        min_loss = np.min(self.val_loss)
        self.predict()
        
        save_all_models = 1
        if min_loss < 0.9 or save_all_models == 1:
            self.save_model()
        save_plot = 0
        self.plot_losses(save_plot = 0)
    
    self.plot_losses_many_runs(save_plot = 0)
    
    
    
def predict(self, subset = 1, verbose = 1):
    def softmax(predicted_output):
        a = np.exp(predicted_output)
        b = np.sum(a,1).reshape(-1,1)
        return a/b
    prediction_model = self.prediction_model
    all_answer_options_intseq = self.data.cache.all_answer_options_intseq
    explain_intseq = self.data.exp_intseq
    questions_intseq = self.data.questions_intseq
    answers = self.data.cache.answers
    indices = self.data.indices
    int_ans = np.array([self.data.convert_to_int(letter) for letter,ans in answers])
    
    
    if subset == 1:
        train_indices,val_indices,test_indices = self.data.indices            
        train_indices = train_indices[0:150]
        val_indices = val_indices[0:150]
        test_indices = test_indices[0:150]
        indices = [train_indices,val_indices,test_indices]
    
    prediction_model.compile(optimizer = 'adam', loss = lambda y_true,y_pred: y_pred, metrics = [keras.metrics.categorical_accuracy])
    all_answer_options_intseq = np.array(all_answer_options_intseq)
    acc = []
    
    for i in range(3):
        ind = indices[i]
        input1 = explain_intseq[ind]
        input2 = questions_intseq[ind]
        input3 = all_answer_options_intseq[:,0,:][ind]
        input4 = all_answer_options_intseq[:,1,:][ind]
        input5 = all_answer_options_intseq[:,2,:][ind]
        input6 = all_answer_options_intseq[:,3,:][ind]
        predicted_output = prediction_model.predict([input1,input2,input3,input4,input5,input6],
                                                    batch_size = 64,
                                                    verbose = verbose)
        predicted_output_softmax = softmax(predicted_output)
        predicted_ans = np.argmax(predicted_output,axis = 1)
        accuracy = np.mean(predicted_ans == int_ans[ind])
        acc.append(accuracy)

    print('train,val,test accuracies: {:.2f}/{:.2f}/{:.2f}'.format(acc[0],acc[1],acc[2]))
    
    
    cache = Struct()
    cache.predicted_output = predicted_output
    cache.predicted_output_softmax = predicted_output_softmax
    cache.predicted_ans = predicted_ans
    cache.int_ans = int_ans
    self.predictions_cache = cache
    self.acc = acc    
    return cache       