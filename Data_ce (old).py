#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:08:30 2018

@author: liyuan
"""
'''
this is the old implementation of character embeddings, originally each sentence/paragraph is converted in its entirety to characters (so each explanation is about 2000 length)
in the new implementation each word in the sentence/paragraph is converted into a list of characters (length 19)
'''


from Data import Data

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
class Data_ce(Data):
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
    char_vocab = None
    char2index = None
    
    def __init__(self):
        pass
       
    def load_raw(self):
        file = open('./wtc_data/questions2.txt',encoding = 'utf-8')
        raw_question = file.readlines()
        raw_question = [self.replacer(text.strip()) for text in raw_question]
        
        file = open('./wtc_data/explanations2.txt','r')
        raw_exp = file.readlines()
        raw_exp = [self.replacer(text.strip()) for text in raw_exp]
        
        self.raw_question = raw_question
        self.raw_exp = raw_exp
    
    
    def get_vocab(self):
        raw_question = self.raw_question
        raw_exp = self.raw_exp
        
        char_vocab = set()
        for text in raw_exp:
            char_vocab = char_vocab | set(text)
        
        for text in raw_question:
            char_vocab = char_vocab | set(text)
            
        char_vocab = sorted(char_vocab)
        char2index = {char:i+1 for i,char in enumerate(char_vocab)} 

        self.char_vocab = char_vocab
        self.char2index = char2index

    def preprocess_question(self):
        raw_question = self.raw_question
        
        questions = []
        answers = []
        all_answer_options_with_questions= []
        answer_indices = []
        
        for text in raw_question:
            raw_question,ans_letter = text.split(' : ')
            
            split_question = self.split_question_and_answers(raw_question)
            split_question = [fragment.strip() for fragment in split_question]
            
            question_part = split_question[0]
            answer_part = split_question[1:]
            answer_index = self.convert_to_int(ans_letter)
            answer_indices.append(answer_index)
            correct_ans_string = answer_part[answer_index]
            ans = [ans_letter,correct_ans_string]
            
            # tokenized as characters
            tokenized_question = list(raw_question)
            questions.append(tokenized_question)
            answers.append(ans)
            all_answer_options_with_questions.append([tokenized_question] + answer_part)
            all_answer_options = [part[1:] for part in all_answer_options_with_questions]            
            
        maxlen_question = max([len(sent) for sent in questions])    
        maxlen_answer = max([max([len(sentence) for sentence in part]) for part in all_answer_options])
        
        cutoff_length = maxlen_question
        questions_intseq = [self.to_intseq(question) for question in questions]
        questions_intseq = pad_sequences(questions_intseq,cutoff_length,value = 0)
        
        answers_words = [sent for option,sent in answers]
        answers_intseq = [self.to_intseq(answer[1]) for answer in answers]
        answers_intseq = pad_sequences(answers_intseq, maxlen_answer,value = 0)
        
        all_answer_options_intseq = [pad_sequences([self.to_intseq(answer) for answer in answers],maxlen_answer,value = 0) for answers in all_answer_options]
        
        wrong_answers = [np.delete(part,index,axis = 0) for part,index in zip(all_answer_options_intseq,answer_indices)]
        
        
        self.questions_intseq = questions_intseq
        self.answers_intseq = answers_intseq
        
        self.cache.questions = questions            
        self.cache.answers = answers
        self.cache.all_answer_options = all_answer_options
        self.cache.all_answer_options_intseq = all_answer_options_intseq
        self.cache.all_answer_options_with_questions = all_answer_options_with_questions            
        self.cache.wrong_answers = wrong_answers  
        
    def preprocess_exp(self):
        """
        make vocab dictionary for all explanations, convert explanations to integer sequence
        """
        with open('./wtc_data/explanations2.txt','r') as file:
            raw_exp = file.readlines()
            exp_tokenized = [list(para.strip()) for para in raw_exp]
            exp_intseq = [self.to_intseq(para) for para in exp_tokenized]
            exp_intseq = pad_sequences(exp_intseq, value = 0)
            
        self.cache.raw_exp = raw_exp
        self.cache.exp_tokenized = exp_tokenized
        self.exp_intseq = exp_intseq
        
    def preprocess_data(self):
        self.load_raw()
        self.get_vocab()
        self.preprocess_exp()
        self.preprocess_question()
        self.get_lengths()
        
        num_train = 1363
        num_val = 150
        num_test = 150
        train_indices,val_indices,test_indices = self.get_shuffled_indices(num_examples = 1663, proportions = [num_train,num_val,num_test])
        self.indices = [train_indices,val_indices,test_indices]
        self.dummy_labels_train = np.array([None]*num_train).reshape(num_train,1)
        self.dummy_labels_val = np.array([None]*num_val).reshape(num_val,1)
        self.answers_intseq2_val = self.sample_wrong_answers()
        
    def replacer(self,raw_sentence):
        raw_sentence = re.sub('[âÂ]','',raw_sentence,flags = re.I) 
        raw_sentence = raw_sentence.replace('\x93','')
        raw_sentence = raw_sentence.replace('\x9d','')
        return raw_sentence
        
    def to_intseq(self,char_list):
        char2index = self.char2index
        intseq = [char2index[char] for char in char_list]
        return intseq
    
    def get_lengths(self):
        super().get_lengths()
        '''because must include padding character'''
        self.lengths.char2index_length = len(self.char2index) + 1
        self.lengths.maxlen_answer = self.answers_intseq.shape[1]
        
    
#%%
        
if __name__ == '__main__':
    temp = Data_ce()
    temp.load_raw()
    temp.get_vocab()
    temp.preprocess_data()
