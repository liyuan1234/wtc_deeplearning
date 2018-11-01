#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:08:30 2018

@author: liyuan
"""

from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import codecs
import keras.backend as K
import time
import datetime
#import config
import re
from Struct import Struct

#%% define class
class Data:
    explanations_path = './wtc_data/explanations2.txt'
    questions_path = './wtc_data/questions2.txt'
    word_embeddings_path = './embeddings/binaries/glove.840B.300d'
#    word_embeddings_path = './embeddings/binaries/glove.6B.50d'
    cache = Struct()
    lengths = Struct()
    indices = []
    embedding_matrix = None
    word2index = None
    reduced_embedding_matrix = None
    reduced_word2index = None
    questions_intseq = None
    exp_intseq = None
    answers_intseq = None
    answers_intseq2_val = None
    dummy_labels_train = None
    dummy_labels_val = None
    raw_questions = None
    exp_vocab = None
    question_vocab = None
    vocab = None
    complete_vocab = None
    
    def __init__(self):
        pass
    
    def __str__(self):
        """
        print attributes
        """
        attr_list = []
        for attribute in dir(self):
            if attribute != '__weakref__':
                if not hasattr(getattr(self,attribute),'__call__') and not '__' in attribute :
                    attribute_value = getattr(self,attribute)
                    if isinstance(attribute_value, (np.ndarray, np.generic)):
                        attribute_value = 'array with shape {}'.format(str(attribute_value.shape))
                    elif isinstance(attribute_value, Struct):
                        attribute_value = 'Struct with {} attributes'.format(len(attribute_value))
                    elif isinstance(attribute_value, list):
                        attribute_value = 'list with {} elements'.format(len(attribute_value))
                    if (attribute_value is not None) and len(attribute_value) > 100:
                        # print in red then revert to white
#                        attribute_value = '\033[33m'+'too long to display...'+'\033[0m'
                        attribute_value = '[{}]'.format(len(attribute_value))
                    line = '{:>30s} : {:<10s}'.format(attribute,str(attribute_value))
                    attr_list.append(line)
        return '\n'.join(attr_list)
    
    def get_vocab(self):
        