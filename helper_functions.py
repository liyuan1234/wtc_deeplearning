#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:16:43 2018

@author: liyuan
"""

from matplotlib import pyplot as plt
import datetime


def plot_loss_history(training_loss,val_loss, save_image = 1, title = ''):
    plt.plot(val_loss, label = 'validation loss')
    plt.plot(training_loss, label = 'training loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch num')
    plt.title(title)
    if save_image == 1:
        timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
        plt.savefig('./images/loss_'+timestamp)