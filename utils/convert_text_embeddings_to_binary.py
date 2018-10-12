#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:18:57 2018

@author: liyuan
"""

from __future__ import print_function

import sys
import codecs
import numpy as np
import time


def main():
    start = time.time()
    embedding_file = '../embeddings/glove.840B.300d.txt'
    print('Loading embeddings file from {}'.format(embedding_file))
    wv, words = load_embeddings(embedding_file)
    print(wv.shape)
    print(len(words))

    out_emb_file, out_vocab_file = embedding_file.replace('.txt', ''), embedding_file.replace('.txt', '.vocab')
    print('Saving binary file to {}'.format(out_emb_file))
    np.save(out_emb_file, wv)

    print('Saving vocabulary file to {}'.format(out_vocab_file))
    with codecs.open(out_vocab_file, 'w', 'utf-8') as f_out:
        for word in words:
            f_out.write(word + '\n')
    print('time taken is {:.2f}'.format(time.time()-start))

def load_embeddings(file_name):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :return: the vocabulary and the word vectors
    """
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        lines = [line.strip() for line in f_in]
    
    embedding_dim = len(lines[0].split()) - 1
    padding_line = ['<pad>'+' 0.0'*embedding_dim]
    lines = padding_line+lines
    words, vectors = zip(*[line.strip().split(' ', 1) for line in lines if len(line.split()) == embedding_dim + 1])
    wv = np.loadtxt(vectors)
    wv = wv.astype(np.float32)

    return wv, words


if __name__ == '__main__':
    main()