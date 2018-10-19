#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:20:48 2018

@author: liyuan
"""

from Deep_qa import Deep_qa
import models

temp = Deep_qa()
temp.load_data()
temp.load_model(models.cnn)
#temp.adapt_embeddings()
temp.run_many_times(self,num_runs = 5,num_iter = 1, learning_rate = 0.001, decay = 0, batch_size = 256, num_epochs = 5,save_plot = 1)