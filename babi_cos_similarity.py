#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:20:45 2018

@author: liyuan
"""
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file

from load_glove_embeddings import load_glove_embeddings

from matplotlib import pyplot as plt
from helper_functions import plot_loss_history

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import LSTM,Input,Add,Dropout,Concatenate,Lambda,Dense, Bidirectional, GRU, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K

import os
import socket

if socket.gethostname() == 'aai-DGX-Station':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


#%% helper functions

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))

def get_wrong_answers(num_train,word_idx,correct_answers,possible_answers = ['bathroom','bedroom','garden','hallway','kitchen','office']):
    
    wrong_answers = [np.random.choice(remove_item(possible_answers,correct_answer)) for correct_answer in correct_answers]
    wrong_answers = [word_idx[word] for word in wrong_answers]
    wrong_answers = keras.utils.to_categorical(wrong_answers,len(word_idx)+1)
    return wrong_answers

def remove_item(mylist,item):
    mynewlist = mylist[:]
    mynewlist.remove(item)
    return mynewlist


#%% define loss functions

def hinge_loss(inputs):
    similarity1,similarity2 = inputs
#    print(similarity1,similarity2)
    hinge_loss = similarity1 - similarity2 - 1
    hinge_loss = -hinge_loss
    loss = K.maximum(0.0,hinge_loss)
    return loss

def _loss_tensor(y_true,y_pred):
    return y_pred

def get_cosine_similarity(input_tensors):
    x,y = input_tensors
    similarity = K.sum(x*y)/get_norm(x)/get_norm(y)
    similarity = K.reshape(similarity,[1,1])
    return similarity

def get_norm(x):
    norm = K.sum(x**2)**0.5
    return norm

#%% start of script
    
load_embeddings = 1
if load_embeddings == 1 or 'word2index' not in vars():
    word2index, embedding_matrix = load_glove_embeddings('./embeddings/glove.6B.50d.txt', embedding_dim=50)

try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
    
    
# 1000 examples
#challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# 10000 examples
challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'


with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)


x_train,xq_train,y_train = x[0:9500],xq[0:9500],y[0:9500]
x_val,xq_val,y_val = x[9500:10000],xq[9500:10000],y[9500:10000]
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

correct_answers = [x[2] for x in train]
correct_answers_train = correct_answers[0:9500]
correct_answers_val = correct_answers[9500:10000]

num_train = len(x_train)
num_val = len(x_val)
num_test_ex = len(test)
wrong_answers = get_wrong_answers(num_train,word_idx,correct_answers_train)
wrong_answers_val = get_wrong_answers(num_val,word_idx,correct_answers_val)




context_length = x.shape[1]
question_length = xq.shape[1]
ans_length = y.shape[1]

#%% set hyperparameters

NUM_HIDDEN_UNITS = 50
num_iter = 20
LEARNING_RATE = 0.0001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)


#%% define model
FC_layer = Dense(NUM_HIDDEN_UNITS)

Cosine_similarity = Lambda(get_cosine_similarity,name = 'Cosine_similarity')

myEmbedding = Embedding(input_dim = vocab_size, output_dim = NUM_HIDDEN_UNITS)

input_explain = Input((context_length,) ,name = 'explanation')
X1 = myEmbedding(input_explain)



input_question = Input((question_length,), name = 'question')
X2 = myEmbedding(input_question)
X2 = LSTM(NUM_HIDDEN_UNITS, name = 'question_representation')(X2)
X2 = Dropout(0.3)(X2)
X2 = RepeatVector(context_length)(X2)

merged = Add()([X1, X2])
question_context_rep = LSTM(NUM_HIDDEN_UNITS)(merged)
question_context_rep = Dropout(0.3)(question_context_rep)

pos_ans = Input((ans_length,))
neg_ans = Input((ans_length,))

pos_ans_rep = FC_layer(pos_ans)
neg_ans_rep = FC_layer(neg_ans)

pos_ans_rep = Dropout(0.3)(pos_ans_rep)
neg_ans_rep = Dropout(0.3)(neg_ans_rep)

pos_similarity = Cosine_similarity([question_context_rep,pos_ans_rep])
neg_similarity = Cosine_similarity([question_context_rep,neg_ans_rep])

loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity])


#%% training

training_model = Model(inputs = [input_explain,input_question,pos_ans,neg_ans],outputs = loss)
training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
print(training_model.summary())

reset_training = 0
if reset_training or 'val_loss' not in vars():
    val_loss = np.array([]) 
if reset_training or 'training_loss' not in vars():
    training_loss = np.array([])

for i in range(num_iter):
    print('running iteration {}...'.format(i))
    #OPTIMIZER = keras.optimizers.RMSprop(lr = 0.0001)
    dummy_labels = np.array([None]*num_train).reshape(num_train,1)
    wrong_answers = get_wrong_answers(num_train,word_idx,correct_answers_train)
    X_train = [x_train,xq_train,y_train,wrong_answers]
    X_val = [x_val,xq_val,y_val,wrong_answers_val]
    history = training_model.fit(x = X_train,y = dummy_labels,batch_size = 256,validation_data = [X_val, dummy_labels[0:500]],epochs = 5)
    val_loss = np.append(val_loss,history.history['val_loss'])
    training_loss = np.append(training_loss,history.history['loss'])
    
plot_loss_history(training_loss,val_loss)
    
#%% test model
    
def vectorize_word(word,word_idx):
    word_vector = np.zeros([1,len(word_idx)+1])
    word_vector[0,word_idx[word]] = 1
    return word_vector
    
possible_answers = ['bathroom','bedroom','garden','hallway','kitchen','office']
test_model = Model(inputs = [input_explain,input_question,pos_ans], outputs = pos_similarity)
results = []
dataset = test
for index in range(5,30):

    correct_word = dataset[index][2]
    
    print(' '.join(dataset[index][0]))
    print(' '.join(dataset[index][1]))
    print(correct_word)
        
    
    predicted_word = None
    predicted_similarity = 0
    for word in vocab:
        test_ans = vectorize_word(word,word_idx)
        similarity = test_model.predict([x[index:index+1],xq[index:index+1],test_ans])
#        if similarity > 0.5 or word == dataset[index][2]:
        if word in possible_answers:
            statement = ('similarity score for '+'{:>10}'.format(word)+ ' is {:>10.5f}').format(similarity[0,0])
            print(statement)
        if similarity > predicted_similarity:
            predicted_word = word
            predicted_similarity = similarity
    print('\npredicted word is :'+predicted_word+'\n')
    input()
    results.append(predicted_word == correct_word)
    
print('\n\n mean correct is {}'.format(np.mean(results)))
