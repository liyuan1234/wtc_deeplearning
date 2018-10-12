#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:15:13 2018

@author: liyuan
"""

import numpy as np
import codecs


    
def load_glove_embeddings(embeddings_path):
    ''' see https://www.quora.com/What-is-a-fast-efficient-way-to-load-word-embeddings-At-present-a-crude-custom-function-takes-about-3-minutes-to-load-the-largest-GloVe-embedding
    '''
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
 
    word2index = {w: i for i, w in enumerate(index2word)}
    embedding_matrix = np.load(embeddings_path + '.npy').astype('float32')
    
    return word2index, embedding_matrix

