#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:16:43 2018

@author: liyuan
"""

from matplotlib import pyplot as plt
import datetime


def plot_loss_history(training_loss,val_loss, save_image = 0, title = ''):
    plt.plot(val_loss, label = 'validation loss')
    plt.plot(training_loss, label = 'training loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch num')
    plt.title(title)
    plt.show()
    if save_image == 1:
        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
        plt.savefig('./images/loss_'+title+'_'+timestamp)
        
        
def save_model_formatted(prediction_model,num_hidden_units):
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
    pooling_type = 'avgpool'
    model_name = 'rnn4_{}_{}_{}.h5py'.format(pooling_type,str(num_hidden_units),timestamp)
    prediction_model.save('./saved_models/' + model_name)    
        
    
def plot_losses_many_runs(loss_cache,title = None,save_fig = 0):
    if title == None:
        title = 'unknown_model'
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
    title = './images/loss_'+title+timestamp
    
    for training_losses,validation_losses in loss_cache:
        plt.plot(training_losses,'r')
        plt.plot(validation_losses,'b')
    plt.plot(training_losses,'r',label = 'training loss')
    plt.plot(validation_losses,'b',label = 'validation loss')  
    plt.legend()
    if save_fig == 1:
        plt.savefig(title) 
        print(title)
    plt.show()
