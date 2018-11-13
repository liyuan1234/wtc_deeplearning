#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:48:10 2018

@author: liyuan
"""



#%%
import re
from keras.preprocessing.sequence import pad_sequences

def replacer(raw_sentence):
    raw_sentence = re.sub('[âÂ]','',raw_sentence,flags = re.I) 
    raw_sentence = raw_sentence.replace('\x93','')
    raw_sentence = raw_sentence.replace('\x9d','')
    return raw_sentence
    
def to_intseq(char_list):
    intseq = [char2index[char] for char in char_list]
    return intseq
    


#%%
file = open('./wtc_data/questions2.txt',encoding = 'utf-8')
raw_question = file.readlines()
raw_question = [list(replacer(text.strip())) for text in raw_question]

file = open('./wtc_data/explanations2.txt','r')
raw_exp = file.readlines()
raw_exp = [list(replacer(text.strip())) for text in raw_exp]

char_vocab = set()
for text in raw_exp:
    char_vocab = char_vocab | set(text)

for text in raw_question:
    char_vocab = char_vocab | set(text)
    
char_vocab = sorted(char_vocab)
char2index = {char:i+1 for i,char in enumerate(char_vocab)}

question_intseq = [to_intseq(text) for text in raw_question]
exp_intseq = [to_intseq(text) for text in raw_exp]

question_intseq = pad_sequences(question_intseq)
exp_intseq = pad_sequences(exp_intseq)



    
#%%

