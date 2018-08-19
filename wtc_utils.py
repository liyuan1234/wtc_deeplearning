#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:17:52 2018

@author: liyuan
"""
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np

def preprocess_exp():
    # make vocab dictionary for all explanations, convert explanations to integer sequence
    file = open('./wtc_data/explanations2.txt','r')
    raw_exp = file.readlines()
    exp_vocab = set()
    for paragraph in raw_exp:
        tokenized_paragraph = nltk.word_tokenize(paragraph)
        exp_vocab = exp_vocab | set(tokenized_paragraph)
    exp_vocab = sorted(exp_vocab)
    exp_vocab_dict = {word:ind+1 for ind,word in enumerate(exp_vocab)}
    
    exp_tokenized = [nltk.word_tokenize(paragraph) for paragraph in raw_exp]
    exp_intseq = [token_sentence_to_intseq(sentence,exp_vocab_dict) for sentence in exp_tokenized]
    exp_intseq = pad_sequences(exp_intseq)
    return exp_vocab,exp_vocab_dict,exp_tokenized,exp_intseq

def token_sentence_to_intseq(sentence,vocab_dict):
    intseq = [vocab_dict[word] for word in sentence]
    return intseq

def convert_to_int(letter):
    if letter == 'A':
        return 0
    if letter == 'B':
        return 1
    if letter == 'C':
        return 2
    if letter == 'D':
        return 3

def replace_unicode_symbols_and_numbers(raw_sentence):
    # replace unicode symbols
    raw_sentence = raw_sentence.replace('\\u00b0C',' deg')
    raw_sentence = raw_sentence.replace('\\u00baC',' deg')
    raw_sentence = raw_sentence.replace('\\u00b0F',' degF')    
    raw_sentence = raw_sentence.replace('\\u00b0',' anglesymb')   
    raw_sentence = raw_sentence.replace('\\"','"')
    raw_sentence = raw_sentence.replace('\\u00f1','n')
    
    # replace numbers with words
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
    

def preprocess_questions(exp_vocab_dict):
    file_dir1 = './wtc_data/questions2.txt'
    file = open(file_dir1,encoding = 'utf-8')
    raw = file.readlines()
    
    # remove newline characters and double quotes
    raw = [text.rstrip().strip('"') for text in raw]
    
    
    # turn question into list of separate words, make separate lists for questions and answers
    questions = []
    answers = []
    for text in raw:
        raw_question,ans = text.split(' : ')
    #    raw_question = raw_question.replace('(A)','AAA')
    #    raw_question = raw_question.replace('(B)','BBB')
    #    raw_question = raw_question.replace('(C)','CCC')
    #    raw_question = raw_question.replace('(D)','DDD')    
    
        # convoluted way to get answer string such that answer contains two parts 
        #'A' and 'sound in a loud classroom' for example.
        splitquestion = raw_question.split('(')
        ans_string = [part.replace(ans+') ','') for part in splitquestion if (ans+')') in part][0]
        ans_string = ans_string.lower()
        ans_string = replace_unicode_symbols_and_numbers(ans_string)
        ans_string = nltk.word_tokenize(ans_string)
        ans = [ans,ans_string]
    
    
        raw_question = raw_question.lower()
        raw_question = replace_unicode_symbols_and_numbers(raw_question)
        
        
        # separate question into a list of words and punctuation
        tokenized_question = nltk.word_tokenize(raw_question)
        questions.append(tokenized_question)
        answers.append(ans)
    
    # make vocab    
    questions_vocab = set()
    for question in questions:
        questions_vocab = questions_vocab | set(question)
    questions_vocab = sorted(questions_vocab)
    questions_vocab_idx = {c: i+1 for i,c in enumerate(questions_vocab)}
    questions_vocab_idx['unk'] = 0
    
    # make each question into a sequence of integers, use unk if word not in list
    questions_intseq = []
    for question in questions:
        token_list = []    
        for word in question:
            try:
                token_list.append(questions_vocab_idx[word])
            except KeyError:
                token_list.append(questions_vocab_idx['unk'])
        questions_intseq.append(token_list)
    
    # calculate some lengths
    maxlen_question = max([len(sent) for sent in questions])    
    vocablen_question = len(questions_vocab)+1
    num_examples = len(raw)
    
    questions_intseq = pad_sequences(questions_intseq,maxlen_question)
    
    # answers_words is a list of each answer, expressed as a tokenized list of that answer sentence
    answers_words = [sent for option,sent in answers]
    
    #convert every word in answers_words to its index (e.g. 'teacher' to 1456)
    answers_intseq = []
    for example in answers_words:
        ind_list = []
        for word in example:
            try:
                ind = exp_vocab_dict[word]
            except KeyError:
                ind = 0
            ind_list.append(ind)
        answers_intseq.append(ind_list)
        
#    answers_intseq = [[questions_vocab_idx[word] for word in example] for example in answers_words ]
    
    
    # convert sequences of numbers to multiclass vector encoding i.e. [400,500,4250] to [0,0,...,1,...,1,...,0,0,0]
    answers_final_form = np.zeros([num_examples,vocablen_question])
    for i in range(num_examples):
        for j in range(len(answers_intseq[i])):
            answers_final_form[i,answers_intseq[i][j]] = 1
    
    
    
    
    
#    # convert answers to 1,2,3,4
#    answers_int = [option for option,_ in answers]
#    answers_int = list([convert_to_int(x) for x in answers_int])
#    answers_onehot = to_categorical(answers_int)
    
    cache = questions_vocab_idx,questions_vocab,questions,answers,answers_intseq
    return questions_intseq, answers_final_form, cache

def get_lengths(questions,exp_intseq,questions_vocab,exp_vocab):
    # calculate some variables
    maxlen_question = max([len(sent) for sent in questions])
    maxlen_exp = max([len(sent) for sent in exp_intseq])
    vocablen_question = len(questions_vocab)+1
    vocablen_exp = len(exp_vocab)+1
    lengths = maxlen_question,maxlen_exp,vocablen_question,vocablen_exp
    return lengths

def preprocess_data():
    """ 
    reads questions2.txt and explanations2.txt and returns questions and explanations in fully processed form, i.e. questions as sequences of numbers, one number for each word, and similarly for explanations
    """
    exp_vocab,exp_vocab_dict,exp_tokenized,exp_intseq = preprocess_exp()
    questions_intseq, answers_final_form, cache = preprocess_questions(exp_vocab_dict)
    questions_vocab_idx,questions_vocab,questions,answers,answers_intseq = cache
    lengths = get_lengths(questions,exp_intseq,questions_vocab,exp_vocab)    
    cache = questions_vocab_idx,questions_vocab,questions,answers,answers_intseq,exp_vocab,exp_vocab_dict,exp_tokenized

    data = questions_intseq,answers_final_form,exp_intseq,lengths,cache
    return data 
    
    

#%%

if __name__ == '__main__':
    data = preprocess_data()
    
