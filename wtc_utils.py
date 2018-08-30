#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:17:52 2018

@author: liyuan
"""
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import keras.backend as K
from load_glove_embeddings import load_glove_embeddings
import time


def preprocess_data():
    """ 
    reads questions2.txt and explanations2.txt and returns questions and explanations in fully processed form, i.e. questions as sequences of numbers, one number for each word, and similarly for explanations
    """

    if 'word2index' not in locals() or globals():    
        global word2index        
        start = time.time()
        word2index, embedding_matrix = load_glove_embeddings('./embeddings/glove.6B.300d.txt', embedding_dim=300) 
        print('time taken to load embeddings: {:.2f}'.format(time.time()-start))
    
    exp_vocab,exp_vocab_dict,exp_tokenized,exp_intseq = preprocess_exp()
    questions_intseq, answers_final_form, cache = preprocess_questions(exp_vocab_dict)
    questions = cache['questions']
    questions_vocab = cache['questions_vocab']
    
    lengths = get_lengths(questions,exp_intseq,questions_vocab,exp_vocab)    
    cache.update({'exp_vocab':exp_vocab, 'exp_vocab_dict':exp_vocab_dict,'exp_tokenized':exp_tokenized})
    data = questions_intseq,answers_final_form,exp_intseq,lengths,cache
    return data 

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
    exp_intseq = [tokenized_sentence_to_intseq(sentence,word2index) for sentence in exp_tokenized]
    exp_intseq = pad_sequences(exp_intseq,value = word2index[''])
    return exp_vocab,exp_vocab_dict,exp_tokenized,exp_intseq

def preprocess_questions(exp_vocab_dict):
    file_dir1 = './wtc_data/questions2.txt'
    file = open(file_dir1,encoding = 'utf-8')
    raw = file.readlines()
    blank_index = word2index['']
    
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
        splitquestion = split_question(raw_question)
        question_part = splitquestion[0]
        answer_part = splitquestion[1:]
        all_answer_options_for_one_question = [process_sentence(sentence) for sentence in answer_part]
        answer_index = convert_to_int(ans_letter)
        answer_indices.append(answer_index)
        correct_ans_string = all_answer_options_for_one_question[answer_index]
        ans = [ans_letter,correct_ans_string]
        
        # separate question into a list of words and punctuation        
        tokenized_question = process_sentence(raw_question)        
        questions.append(tokenized_question)
        answers.append(ans)
        all_answer_options_with_questions.append([tokenized_question] + all_answer_options_for_one_question)
        all_answer_options = [part[1:] for part in all_answer_options_with_questions]
        
    # make vocab    
    questions_vocab = set()
    for question in questions:
        questions_vocab = questions_vocab | set(question)
    questions_vocab = sorted(questions_vocab)
    questions_vocab_idx = {c: i+1 for i,c in enumerate(questions_vocab)}
    questions_vocab_idx['unk'] = 0
    
    # calculate some lengths
    maxlen_question = max([len(sent) for sent in questions])    
    vocablen_question = len(word2index)+1
    num_examples = len(raw)    
    maxlen_answer = max([max([len(sentence) for sentence in part]) for part in all_answer_options])
    
    # make each question into a sequence of integers, use unk if word not in list
    questions_intseq = convert_to_intseq(questions,word2index)
    questions_intseq = pad_sequences(questions_intseq,maxlen_question,value = blank_index)
    
    # answers_words is a list of each answer, expressed as a tokenized list of that answer sentence
    #convert every word in answers_words to its index (e.g. 'teacher' to 1456)    
    answers_words = [sent for option,sent in answers]
    answers_intseq = convert_to_intseq(answers_words,word2index)
    answers_intseq = pad_sequences(answers_intseq, maxlen_answer,value = blank_index) 
    
    '''
    all_answer_options is a list of tokenized answers e.g. [['large','leaves'],[shallow','roots'],...]
    all_answer_options_intseq is the same list padded and converted to integer representations
    e.g. [[0,0,0,...,]]
    '''
    all_answer_options_intseq = [[tokenized_sentence_to_intseq(sentence,word2index) for sentence in part] for part in all_answer_options]
    all_answer_options_intseq = [pad_sequences(part,maxlen_answer,value = blank_index) for part in all_answer_options_intseq]
    
    wrong_answers = [np.delete(part,index,axis = 0) for part,index in zip(all_answer_options_intseq,answer_indices)]
            
    
    # convert sequences of numbers to multiclass vector encoding i.e. [400,500,4250] to [0,0,...,1,...,1,...1,...,0,0,0]
    answers_final_form = np.zeros([num_examples,vocablen_question])
    for i in range(num_examples):
        for j in range(len(answers_intseq[i])):
            answers_final_form[i,answers_intseq[i][j]] = 1
    
    cache = {'questions_vocab_idx':questions_vocab_idx,
             'questions_vocab':questions_vocab,
             'questions':questions,
             'answers':answers,
             'answers_intseq':answers_intseq,
             'all_answer_options_with_questions':all_answer_options_with_questions,
             'all_answer_options':all_answer_options,
             'all_answer_options_intseq':all_answer_options_intseq,
             'wrong_answers':wrong_answers}
    return questions_intseq, answers_final_form, cache

def remaining_indices(index):
    remaining_indices = [0,1,2,3]
    remaining_indices.remove(index)
    return remaining_indices

def process_sentence(sentence):
    ''' takes in a sentence string and returns a list of separate words'''
    
    word_list = replace_unicode_symbols_and_numbers(sentence.lower())
    word_list = nltk.word_tokenize(word_list)
    return word_list

def split_question(question):
    question = question.replace('(A)','(XXX)')
    question = question.replace('(B)','(XXX)')
    question = question.replace('(C)','(XXX)')
    question = question.replace('(D)','(XXX)')
    splitquestion = question.split('(XXX)')
    
    # this is to make all answers have 4 options, since some questions only have 3 options, 
    # split question contains the question and the four answers
    while len(splitquestion) < 5:
        splitquestion.append('')
    return splitquestion
        
    
    
def convert_to_intseq(tokenized_word_set,vocab_index):
    intseq = []
    for sentence in tokenized_word_set:
        sentence_intseq = tokenized_sentence_to_intseq(sentence,vocab_index)
        intseq.append(sentence_intseq)
    return intseq

def tokenized_sentence_to_intseq(sentence,vocab_index):
    """
    e.g. ['I','like','eggs'] to [41,117,5130]
    """
    intseq = []
    for word in sentence:
        try:
            intseq.append(vocab_index[word.lower()])
        except KeyError:
            intseq.append(vocab_index['unk'])    
    return intseq

def remove_answer_labels(sentence):
    sentence = sentence.replace('A)','')
    sentence = sentence.replace('B)','')
    sentence = sentence.replace('C)','')
    sentence = sentence.replace('D)','')
    return sentence

def get_lengths(questions,exp_intseq,questions_vocab,exp_vocab):
    # calculate some variables
    maxlen_question = max([len(sent) for sent in questions])
    maxlen_exp = max([len(sent) for sent in exp_intseq])
    vocablen_question = len(word2index)
    vocablen_exp = len(word2index)
    lengths = maxlen_question,maxlen_exp,vocablen_question,vocablen_exp
    return lengths

def convert_to_int(letter):
    if letter == 'A':
        return 0
    if letter == 'B':
        return 1
    if letter == 'C':
        return 2
    if letter == 'D':
        return 3

def convert_to_letter(index):
    if index == 0:
        return 'A'
    if index == 1:
        return 'B'
    if index == 2:
        return 'C'
    if index == 3:
        return 'D'
    
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

def sample_wrong_answers(wrong_answers):
    answers_intseq2 = [part[np.random.randint(len(part))] for part in wrong_answers]
    answers_intseq2 = np.array(answers_intseq2)
    return answers_intseq2

#%%

if __name__ == '__main__':   
    data = preprocess_data()
# unpack data
    questions_intseq,answers_final_form,explain_intseq,lengths,cache = data
    maxlen_question,maxlen_explain,vocablen_question,vocablen_explain = lengths
    answers_intseq = cache['answers_intseq']
    wrong_answers = cache['wrong_answers']  
