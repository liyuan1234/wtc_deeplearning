#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:48:10 2018

@author: liyuan
"""

temp = training_model.get_weights()
for i in range(len(temp)):
    print(temp[i].shape)