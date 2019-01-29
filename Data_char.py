#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:20:25 2018

@author: liyuan
"""

'''
Data_char is prepared by converting each word into a list of characters e.g. explanation[0] contains 200 words, this is converted to 270 (max length of explanation) lists of length 19 (max char in word) integer sequence
the result is a 1663x270x19 array
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
class Data_char(Data):
    explanations_path = './data/explanations2.txt'
    questions_path = './data/questions2.txt'
    word_embeddings_path = './embeddings/binaries/glove.840B.300d'
#    word_embeddings_path = './embeddings/binaries/glove.6B.50d'
    cache = Struct()
    lengths = Struct()
    indices = []
    questions_intseq = None
    exp_intseq = None
    answers_intseq = None
    answers_intseq2_val = None
    dummy_labels_train = None
    dummy_labels_val = None
    raw_questions = None
    char_vocab = None
    char2index = None
    
    
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
    
    def preprocess_data(self):
        """ 
        reads questions2.txt and explanations2.txt and returns questions and explanations in fully processed form, i.e. questions as sequences of numbers, one number for each word, and similarly for explanations
        """
        self.load_raw()
        self.get_vocab()
        self.preprocess_exp()
        self.preprocess_questions()
        
        num_train = 1363
        num_val = 150
        num_test = 150
        train_indices,val_indices,test_indices = self.get_shuffled_indices(num_examples = 1663, proportions = [num_train,num_val,num_test])
        self.indices = [train_indices,val_indices,test_indices]
        self.dummy_labels_train = np.array([None]*num_train).reshape(num_train,1)
        self.dummy_labels_val = np.array([None]*num_val).reshape(num_val,1)
        self.answers_intseq2_val = self.sample_wrong_answers()
        
        
        self.exp_intseq = np.array(self.exp_intseq)
        self.questions_intseq = np.array(self.questions_intseq)
        self.answers_intseq = np.array(self.answers_intseq)
        self.get_lengths()


    def load_raw(self):
        file = open('./data/questions2.txt',encoding = 'utf-8')
        raw_question = file.readlines()
        raw_question = [self.replacer(text.strip()) for text in raw_question]
        
        file = open('./data/explanations2.txt','r')
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
        
        raw = raw_question+raw_exp
        raw = [nltk.tokenize.word_tokenize(sentence) for sentence in raw]
        max_char_in_word = max([max([len(list(word)) for word in sentence]) for sentence in raw])

        self.char_vocab = char_vocab
        self.char2index = char2index
        self.lengths.max_char_in_word = max_char_in_word


    def replacer(self,raw_sentence):
        raw_sentence = re.sub('[âÂ]','',raw_sentence,flags = re.I) 
        raw_sentence = raw_sentence.replace('\x93','')
        raw_sentence = raw_sentence.replace('\x9d','')
        return raw_sentence
        
    def preprocess_exp(self):
        """
        make vocab dictionary for all explanations, convert explanations to integer sequence
        """
        raw_exp = self.raw_exp
        raw_exp = [self.replace_text_in_braces(line) for line in raw_exp]
        exp_tokenized = [nltk.word_tokenize(paragraph) for paragraph in raw_exp]
        exp_intseq = self.all_examples_to_intseq(self.pad(exp_tokenized))
        
        self.cache.raw_exp = raw_exp
        self.cache.exp_tokenized = exp_tokenized
        self.exp_intseq = exp_intseq
            
    def preprocess_questions(self):
        filepath = self.questions_path
        
        raw_question = self.raw_question
        blank_index = 0
        
        # remove newline characters and double quotes
        raw_question = [text.rstrip().strip('"') for text in raw_question]
        
        # turn question into list of separate words, make separate lists for questions and answers
        questions = []
        answers = []
        answer_options_all_questions_with_questions= []
        answer_indices = []
        for text in raw_question:
            raw_question,ans_letter = text.split(' : ')
       
            # correct_answer_string contains two parts, the letter answer and the answer string
            #'A' and 'sound in a loud classroom' for example.
            split_question = self.split_question_and_answers(raw_question)
            question_part = split_question[0]
            answer_part = split_question[1:]
            answer_options_all_questions_for_one_question = [self.process_sentence(sentence) for sentence in answer_part]
            answer_index = self.convert_to_int(ans_letter)
            answer_indices.append(answer_index)
            correct_ans_string = answer_options_all_questions_for_one_question[answer_index]
            ans = [ans_letter,correct_ans_string]
            
            # separate question into a list of words and punctuation        
            tokenized_question = self.process_sentence(raw_question)        
            questions.append(tokenized_question)
            answers.append(ans)
            answer_options_all_questions_with_questions.append([tokenized_question] + answer_options_all_questions_for_one_question)
            answer_options_all_questions = [part[1:] for part in answer_options_all_questions_with_questions]
                
            
            
        # calculate some lengths
        maxlen_question = max([len(sent) for sent in questions])    
        maxlen_answer = max([max([len(sentence) for sentence in part]) for part in answer_options_all_questions])
        
        # make each question into a sequence of integers, use unk if word not in list
        cutoff_length = 150
        questions_intseq = self.all_examples_to_intseq(self.pad(questions, cutoff_length = cutoff_length))
        # note: didn't do padding here

        
        # answers_words is a list of each answer, expressed as a tokenized list of that answer sentence
        #convert every word in answers_words to its index (e.g. 'teacher' to 1456)    
        answers_words = [sent for option,sent in answers]
        answers_intseq = self.all_examples_to_intseq(self.pad(answers_words,maxlen_answer))
        # note: didn't do padding here
        
        '''
        answer_options_all_questions is a list of tokenized answers e.g. [['large','leaves'],['shallow','roots'],...]
        all_answer_options_intseq is the same list padded and converted to integer representations
        e.g. [[0,0,0,...,]]
        '''
        
        padded_answer_options = [self.pad(x,maxlen_answer) for x in answer_options_all_questions]
        all_answer_options_intseq = [self.all_examples_to_intseq(answer_options) for answer_options in padded_answer_options]
        
        wrong_answers = [np.delete(part,index,axis = 0) for part,index in zip(all_answer_options_intseq,answer_indices)]
                
        self.questions_intseq = questions_intseq
        self.answers_intseq = answers_intseq
        
        self.raw_question = raw_question
        self.cache.questions = questions            
        self.cache.answers = answers
        self.cache.answer_options_all_questions = answer_options_all_questions
        self.cache.all_answer_options_intseq = all_answer_options_intseq
        self.cache.answer_options_all_questions_with_questions = answer_options_all_questions_with_questions            
        self.cache.wrong_answers = wrong_answers            
            
            


    def to_intseq(self,word):
        char2index = self.char2index
        char2index['`'] = 0
        intseq = [char2index[char] for char in list(word)]
        
        raise_message = 0
        if raise_message and '`' in word:
            print('` detected...' ,end = '')
            print(word)
        return intseq
        
        
    def all_examples_to_intseq(self,all_examples):
        '''expects explanations or questions in tokenized form i.e. each explanation paragraph is tokenized into words
        '''
        max_char_in_word = self.lengths.max_char_in_word
        
        intseq = []
        for words in all_examples:
            intseq_words = [self.to_intseq(word) for word in words]
            intseq_words = pad_sequences(intseq_words,max_char_in_word)
            intseq.append(intseq_words)
        return intseq
        

    def get_lengths(self):
        '''because must include padding character'''
        self.lengths.maxlen_question = max([len(sent) for sent in self.questions_intseq])
        self.lengths.maxlen_raw_question = max([len(sent) for sent in self.cache.questions])
        self.lengths.maxlen_exp = max([len(sent) for sent in self.exp_intseq])
        self.lengths.num_examples = len(self.cache.questions)           
        self.lengths.char2index_length = len(self.char2index) + 1
        self.lengths.maxlen_answer = self.answers_intseq.shape[1]
     
        
    def pad(self,tokenized,maxlen = None,cutoff_length = None):
        '''
        pads all examples with '' to same length to facilitate processing
        inputs are tokenized sentences/paragraphs of varying lengths
        '''
        if maxlen == None:
            maxlen = max([len(x) for x in tokenized])
        if cutoff_length == None:
            cutoff_length = maxlen
        padded = [['']*(maxlen - len(sentence)) + sentence for sentence in tokenized]
        padded = [sentence[maxlen-cutoff_length:] for sentence in padded]
        return padded


    def sample_wrong_answers(self):        
        answers_intseq2 = [part[np.random.randint(len(part))] for part in self.cache.wrong_answers]
        answers_intseq2 = np.array(answers_intseq2)
        return answers_intseq2

    
#    def foo(self):
#        exp_tokenized = self.cache.exp_tokenized
#        question_tokenized = self.cache.questions
#        
#        vocab = set()
#        for paragraph in exp_tokenized
        
        
#            exp_vocab = set()
#            for paragraph in raw_exp:
#                tokenized_paragraph = nltk.word_tokenize(paragraph)
#                exp_vocab = exp_vocab | set(tokenized_paragraph)
#            exp_vocab = sorted(exp_vocab)
#            exp_vocab_dict = {word:ind+1 for ind,word in enumerate(exp_vocab)}

#%% example
        
if __name__ == '__main__':
    temp = Data_char()
    temp.preprocess_data()
    print(temp)
