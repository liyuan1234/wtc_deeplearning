#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:32:14 2018

@author: liyuan
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import keras.backend as K
import keras
import copy

from loss_functions import hinge_loss, _loss_tensor


def count_params(self):
    model = self.training_model
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    untrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, untrainable_count

def save_model(self, model = None, file_path = None):
    if model == None:
        model = self.training_model
    
    if file_path == None:
        title = self.get_formatted_title()
        filepath = './saved_models/' + title + '.h5'
        model.save('./saved_models/last_model.h5')
        
    model.save(filepath)
    
def save_obj(self):
    ''' pickle doesn't work with keras models, so use this hack to save object and keras model separately
    Usage:
        temp.save_obj()
    '''
    title = self.get_formatted_title()
    fp1 = './pickled/objects/'+title+'.p'
    fp2 = './pickled/training_models/'+title+'.h5'
    fp3 = './pickled/prediction_models/'+title+'.h5'
    
    f1 = open(fp1,'wb')
    f2 = open(fp2,'wb')
    f3 = open(fp3,'wb')

    self.training_model.save(fp2)
    self.prediction_model.save(fp3)
    
    training_model = self.training_model
    prediction_model = self.prediction_model
    self.training_model = None
    self.prediction_model = None
    pickle.dump(self,f1)
    self.training_model = training_model
    self.prediction_model = prediction_model
    
    f1.close()
    f2.close()
    f3.close()  
    
    print('saved object to /pickled ...')
    
def load_obj(file):
    '''
    Usage:
        temp = Deep_qa.load('word(cnn)_10units_0.5thres_0.255loss_0.90_0.66_0.63acc')
        
    there seems to be a bug when loading a model that uses lambda layers with custom functions
    see https://github.com/keras-team/keras/issues/5298
    workaround is to pass whatever that is missing through custom_object argument when calling load_model
    '''
    
    stem = './pickled'
    fp1 = './pickled/objects/' + file + '.p'
    fp2 = './pickled/training_models/' + file + '.h5'
    fp3 = './pickled/prediction_models/' + file + '.h5'
    obj = pickle.load(open(fp1,'rb'))
    custom_objects = {'hinge_loss':hinge_loss, '_loss_tensor':_loss_tensor}
    training_model = keras.models.load_model(fp2, custom_objects = custom_objects)
    prediction_model = keras.models.load_model(fp3, custom_objects = custom_objects)
    obj.training_model = training_model
    obj.prediction_model = prediction_model
    return obj
    
    
def get_formatted_title(self):
    if self.acc == None:
        raise ValueError("Can't get title, acc not known, run predict first!")
    
    flag = self.flag
    model_flag = self.model_flag
    threshold = self.model_params.threshold
    train_acc,val_acc,test_acc = self.acc
    val_loss = np.min(self.val_loss)
    units = self.model_params.units
    units_char = self.model_params.units_char
    training_model = self.training_model
    
    if units_char == None:
        units_char = ''
    else:
        units_char = '({})'.format(units_char)
    
    title = '{:s}({:s})_{}{}units_{:.1f}thres_{:.3f}loss_{:.2f}_{:.2f}_{:.2f}acc'.format(flag, model_flag, units, units_char, threshold, val_loss, train_acc, val_acc, test_acc)  
    return title
    
def plot_losses(self,losses = None, save_plot = 0, maximize_yaxis = 1):
    if losses == None:
        training_loss = self.training_loss
        val_loss = self.val_loss
    else:
        training_loss,val_loss = losses
    
    plt.plot(val_loss, label = 'validation loss')
    plt.plot(training_loss, label = 'training loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch num')
    if maximize_yaxis == 1:
        plt.ylim([0,max(val_loss)+0.1])
        
    fig_title = '{}({})_{}units'.format(self.flag,self.model_flag,self.model_params.units)    
    plt.title(fig_title)
    if save_plot == 1:
        title = self.get_formatted_title()
        filepath = './images/'+title+'.png'
        plt.savefig(filepath)
        print('saved figure to {}'.format(filepath))
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
    plt.ylim([0,max(val_loss)+0.1])        
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
        
def save_losses(self,title = None):
    if title == None:
        title = self.get_formatted_title()
    file = open('./pickled/{}.pickle'.format(title),'wb')
    pickle.dump(temp.loss_cache,file)
    file.close()
    
    

def calculate_loss(self, flag = 'val',print_list = 0):
    ''' calculate loss for train, val or test set
    I find that training loss reported at end of epoch is usually lower than training loss from predict method - overfitting to particular set of answers happen during training  
    '''
    train_indices,val_indices,test_indices = self.data.indices        
    if flag == 'train':
        ind = train_indices
    elif flag == 'val':
        ind = val_indices
    elif flag == 'test':
        ind = test_indices
    else:
        print('invalid flag.')
    exp = self.data.exp_intseq[ind]
    question = self.data.questions_intseq[ind]
    ans1 = self.data.answers_intseq[ind]
    ans2 = self.data.answers_intseq2_val[ind]
    loss = self.training_model.predict([exp,question,ans1,ans2],batch_size = 64,verbose = 0)
    
    if print_list == 1:
        print(loss[1:10])
    print(flag+' loss: {:.4f}'.format(np.mean(loss)))
    return loss

def printLosses(self, print_list = 0):
    
    check_indices = 0
    if check_indices == 1:
        train_indices,val_indices,test_indices = self.data.indices        
        print(train_indices[0:5])
        print(val_indices[0:5])
        print(test_indices[0:5])
        
    
    self.calculate_loss('train', print_list)
    self.calculate_loss('val', print_list)
    self.calculate_loss('test', print_list)