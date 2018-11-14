#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:20:25 2018

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
    
    def preprocess_data(self):
        """ 
        reads questions2.txt and explanations2.txt and returns questions and explanations in fully processed form, i.e. questions as sequences of numbers, one number for each word, and similarly for explanations
        """
        self.load_glove_embeddings()
        self.get_reduced_embeddings()
        self.preprocess_exp()
        self.preprocess_questions()
        self.get_lengths()
        
        num_train = 1363
        num_val = 150
        num_test = 150
        train_indices,val_indices,test_indices = self.get_shuffled_indices(num_examples = 1663, proportions = [num_train,num_val,num_test])
        self.indices = [train_indices,val_indices,test_indices]
        self.dummy_labels_train = np.array([None]*num_train).reshape(num_train,1)
        self.dummy_labels_val = np.array([None]*num_val).reshape(num_val,1)
        self.answers_intseq2_val = self.sample_wrong_answers()
        
    
    def load_glove_embeddings(self,word_embeddings_path = None):
        ''' see https://www.quora.com/What-is-a-fast-efficient-way-to-load-word-embeddings-At-present-a-crude-custom-function-takes-about-3-minutes-to-load-the-largest-GloVe-embedding
        '''
        start_time = time.time()
        
        if word_embeddings_path == None:
            word_embeddings_path = self.word_embeddings_path
        
        if self.embedding_matrix is None:
            with codecs.open(word_embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
                index2word = [line.strip() for line in f_in]
         
            word2index = {w: i+1 for i, w in enumerate(index2word)}
            
            embedding_matrix = np.load(word_embeddings_path + '.npy').astype('float32')
            #add a row of zeros
            blank_row = np.zeros([1,embedding_matrix.shape[1]])
            embedding_matrix = np.vstack([blank_row,embedding_matrix])
            print('time taken to load embeddings is {time:.2f}'.format(time = time.time() - start_time)) 
            
            self.word2index = word2index
            self.embedding_matrix = embedding_matrix    
    
    def get_reduced_embeddings(self):
        embedding_matrix = self.embedding_matrix
        word2index = self.word2index

        with open('./wtc_data/explanations2.txt','r') as file:
            raw_exp = file.readlines()
            raw_exp = [self.replace_text_in_braces(line) for line in raw_exp]
            exp_tokenized = [self.process_sentence(line) for line in raw_exp]
        #    exp_tokenized = [nltk.word_tokenize(paragraph) for paragraph in raw_exp]
            
        exp_vocab = set()
        for paragraph in exp_tokenized:
            exp_vocab = exp_vocab | set(paragraph)
                    
        with open('./wtc_data/questions2.txt',encoding = 'utf-8') as file:
            raw = file.readlines()
            raw = [text.rstrip().strip('"') for text in raw]
            questions_tokenized = [self.process_sentence(line) for line in raw]
            
        question_vocab = set()    
        for paragraph in questions_tokenized:
            question_vocab = question_vocab | set(paragraph)
        
        complete_vocab = question_vocab | exp_vocab 
        glove_vocab = set(word2index.keys())
        vocab = complete_vocab & glove_vocab 
        vocab = vocab | {'unk'}
        
        ''' vocab is an alphabetical sorted (symbols, capitalized words, uncapitalized) list of vocab used that is also found in glove'''
        vocab = list(sorted(vocab))
        n_words = len(vocab)
        reduced_embedding_matrix = np.zeros([n_words+1,embedding_matrix.shape[1]])
        
        for i in range(n_words):
            index = word2index[vocab[i]]
            reduced_embedding_matrix[i+1,:] = embedding_matrix[index,:]
            
        reduced_word2index = {w:i+1 for i,w in enumerate(vocab)}
        
        self.reduced_embedding_matrix = reduced_embedding_matrix
        self.reduced_word2index = reduced_word2index
        
        self.exp_vocab = exp_vocab
        self.question_vocab = question_vocab
        self.complete_vocab = complete_vocab
        self.vocab = vocab

        
    def preprocess_exp(self):
        """
        make vocab dictionary for all explanations, convert explanations to integer sequence
        """
        with open('./wtc_data/explanations2.txt','r') as file:
            raw_exp = file.readlines()
            raw_exp = [self.replace_text_in_braces(line) for line in raw_exp]
            exp_tokenized = [nltk.word_tokenize(paragraph) for paragraph in raw_exp]
            exp_intseq = [self.tokenized_sentence_to_intseq(sentence,self.reduced_word2index) for sentence in exp_tokenized]
            exp_intseq = pad_sequences(exp_intseq,value = 0)
        
        self.cache.raw_exp = raw_exp
        self.cache.exp_tokenized = exp_tokenized
        self.exp_intseq = exp_intseq


    def tokenize_exp(self):
        with open('./wtc_data/explanations2.txt','r') as file:
            raw_exp = file.readlines()
            raw_exp = [self.replace_text_in_braces(line) for line in raw_exp]
            exp_tokenized = [nltk.word_tokenize(paragraph) for paragraph in raw_exp]
        
        self.cache.raw_exp = raw_exp
        self.cache.exp_tokenized = exp_tokenized
            

    def preprocess_questions(self):
        filepath = self.questions_path
        reduced_word2index = self.reduced_word2index
        
        with open(filepath,encoding = 'utf-8') as file:
            raw = file.readlines()
            blank_index = 0
            
            # remove newline characters and double quotes
            raw = [text.rstrip().strip('"') for text in raw]
            
            # turn question into list of separate words, make separate lists for questions and answers
            questions = []
            answers = []
            all_answer_options_with_questions= []
            answer_indices = []
            for text in raw:
                raw_question,ans_letter = text.split(' : ')
           
                # correct_answer_string contains two parts, the letter answer and the answer string
                #'A' and 'sound in a loud classroom' for example.
                split_question = self.split_question_and_answers(raw_question)
                question_part = split_question[0]
                answer_part = split_question[1:]
                all_answer_options_for_one_question = [self.process_sentence(sentence) for sentence in answer_part]
                answer_index = self.convert_to_int(ans_letter)
                answer_indices.append(answer_index)
                correct_ans_string = all_answer_options_for_one_question[answer_index]
                ans = [ans_letter,correct_ans_string]
                
                # separate question into a list of words and punctuation        
                tokenized_question = self.process_sentence(raw_question)        
                questions.append(tokenized_question)
                answers.append(ans)
                all_answer_options_with_questions.append([tokenized_question] + all_answer_options_for_one_question)
                all_answer_options = [part[1:] for part in all_answer_options_with_questions]
                    
                
                
            # calculate some lengths
            maxlen_question = max([len(sent) for sent in questions])    
            maxlen_answer = max([max([len(sentence) for sentence in part]) for part in all_answer_options])
            
            # make each question into a sequence of integers, use unk if word not in list
            cutoff_length = 150
            questions_intseq = self.convert_to_intseq(questions,reduced_word2index)
            questions_intseq = pad_sequences(questions_intseq,cutoff_length,value = blank_index)
            
            # answers_words is a list of each answer, expressed as a tokenized list of that answer sentence
            #convert every word in answers_words to its index (e.g. 'teacher' to 1456)    
            answers_words = [sent for option,sent in answers]
            answers_intseq = self.convert_to_intseq(answers_words,reduced_word2index)
            answers_intseq = pad_sequences(answers_intseq, maxlen_answer,value = blank_index) 
            
            '''
            all_answer_options is a list of tokenized answers e.g. [['large','leaves'],['shallow','roots'],...]
            all_answer_options_intseq is the same list padded and converted to integer representations
            e.g. [[0,0,0,...,]]
            '''
            all_answer_options_intseq = [[self.tokenized_sentence_to_intseq(sentence,reduced_word2index) for sentence in part] for part in all_answer_options]
            all_answer_options_intseq = [pad_sequences(part,maxlen_answer,value = blank_index) for part in all_answer_options_intseq]
            
            wrong_answers = [np.delete(part,index,axis = 0) for part,index in zip(all_answer_options_intseq,answer_indices)]
                    
            self.questions_intseq = questions_intseq
            self.answers_intseq = answers_intseq
            
            self.raw_questions = raw
            self.cache.questions = questions            
            self.cache.answers = answers
            self.cache.all_answer_options = all_answer_options
            self.cache.all_answer_options_intseq = all_answer_options_intseq
            self.cache.all_answer_options_with_questions = all_answer_options_with_questions            
            self.cache.wrong_answers = wrong_answers            
            
            

  
    
    
    def get_remaining_indices(self,index):
        remaining_indices = [0,1,2,3]
        remaining_indices.remove(index)
        return remaining_indices

    
    def split_question_and_answers(self,question):
        question = question.replace('(A)','(XXX)')
        question = question.replace('(B)','(XXX)')
        question = question.replace('(C)','(XXX)')
        question = question.replace('(D)','(XXX)')
        split_question = question.split('(XXX)')
        
        # this is to make all answers have 4 options, since some questions only have 3 options, 
        # split question contains the question and the four answers
        while len(split_question) < 5:
            split_question.append('')
        return split_question
            
        
        
    def convert_to_intseq(self,tokenized_word_set,vocab_index):
        intseq = []
        for sentence in tokenized_word_set:
            sentence_intseq = self.tokenized_sentence_to_intseq(sentence,vocab_index)
            intseq.append(sentence_intseq)
        return intseq
    
    def tokenized_sentence_to_intseq(self,sentence,vocab_index):
        """
        e.g. ['I','like','eggs'] to [41,117,5130]
        """
        intseq = []
        for word in sentence:
            try:
                intseq.append(vocab_index[word])
            except KeyError:
                intseq.append(vocab_index['unk'])    
        return intseq

    def get_lengths(self):
        try:
            self.lengths.maxlen_question = max([len(sent) for sent in self.questions_intseq])
            self.lengths.maxlen_raw_question = max([len(sent) for sent in self.cache.questions])
            self.lengths.maxlen_exp = max([len(sent) for sent in self.exp_intseq])
            self.lengths.word2index_length = len(self.word2index)
            self.lengths.num_examples = len(self.cache.questions)
#            self.lengths.maxlen_answer = self.answers_intseq.shape[1]
        except TypeError:
            print('encountered TypeError in get_lengths!')
    
    def convert_to_int(self,letter):
        if letter == 'A':
            return 0
        if letter == 'B':
            return 1
        if letter == 'C':
            return 2
        if letter == 'D':
            return 3
    
    def convert_to_letter(self,index):
        if index == 0:
            return 'A'
        if index == 1:
            return 'B'
        if index == 2:
            return 'C'
        if index == 3:
            return 'D'

    def process_sentence(self,sentence):
        ''' takes in a sentence string and returns a list of separate words'''
        
        word_list = self.replace_unicode_symbols_and_numbers(sentence)
        word_list = nltk.word_tokenize(word_list)
        return word_list

    def replace_text_in_braces(self,line):
        def replacer(m):
            final_str = m.group().replace(';','and')
            final_str = final_str.replace('(','')
            final_str = final_str.replace(')','')
            return final_str
        
        part = '(;[^();]+)'
        regex = '\(([^();]+);([^();]+){}?{}?{}?{}?{}?{}?{}?{}?{}?{}?\)'.format(part,part,part,part,part,part,part,part,part,part)
        
        return re.sub(regex,replacer,line) 



    def replace_unicode_symbols_and_numbers(self,raw_sentence):
        def round_offer(m):
            integer = int(float((m.group())))
            int_string = str(integer)
            return int_string
        
        # deal with some weird formatting
        raw_sentence = re.sub('[âÂ]','',raw_sentence,flags = re.I)        
        raw_sentence = re.sub(r'([-.\d]+)[°º]?[Cc]\b',r'\1 degree celsius ',raw_sentence, flags = re.I)
        raw_sentence = re.sub(r'([-.\d]+)[°º]?[Kk]\b',r'\1 degree kelvin',raw_sentence, flags = re.I)
        raw_sentence = re.sub(r'([-.\d]+)[°º]?[Ff]\b',r'\1 degree fahrenheit',raw_sentence, flags = re.I)
        raw_sentence = re.sub(r'[°º]',r' degree',raw_sentence)
        raw_sentence = re.sub('([-\d]+\.[\d]+)',round_offer,raw_sentence)
        
        
            
        
        
        # replace numbers with words
        replace_numbers = 0
        if replace_numbers:
            raw_sentence = raw_sentence.replace('0','zero ')
            raw_sentence = raw_sentence.replace('1','one ')
            raw_sentence = raw_sentence.replace('2','two ')
            raw_sentence = raw_sentence.replace('3','three ')
            raw_sentence = raw_sentence.replace('4','four ')
            raw_sentence = raw_sentence.replace('5','five ')
            raw_sentence = raw_sentence.replace('6','six ')
            raw_sentence = raw_sentence.replace('7','seven ')
            raw_sentence = raw_sentence.replace('8','eight ')
            raw_sentence = raw_sentence.replace('9','nine ')
        return raw_sentence
    
    def sample_wrong_answers(self):        
        answers_intseq2 = [part[np.random.randint(len(part))] for part in self.cache.wrong_answers]
        answers_intseq2 = np.array(answers_intseq2)
        return answers_intseq2
    
    def get_shuffled_indices(self,num_examples = None, proportions = [1463,100,100]):
        if num_examples == None:
            num_examples = self.lengths.num_examples
        
        np.random.seed(1)
        num_train,num_val,num_test = proportions
        shuffled_indices = np.arange(num_examples)
        np.random.shuffle(shuffled_indices)
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train+num_val]
        test_indices = shuffled_indices[num_train+num_val : num_train+num_val+num_test]
        return train_indices,val_indices,test_indices    
    
    def get_unique_word_indics(self):
        unique_question_indices = np.unique(np.array(self.questions_intseq))
        unique_exp_indices = np.unique(np.array(self.exp_intseq))
        
        unique_indices = np.unique(np.concatenate([unique_question_indices,unique_exp_indices]))
        return unique_indices
    
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
    temp = Data()
    temp.preprocess_data()