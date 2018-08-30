#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:50:04 2018

@author: liyuan
"""

import time
start_time = time.time()

import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Input,Dropout,Reshape,Add,Lambda,Concatenate,Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
import nltk
from load_glove_embeddings import load_glove_embeddings
import tensorflow as tf
import numpy as np

from loss_functions import hinge_loss, _loss_tensor, get_cosine_similarity, get_norm

from wtc_utils import preprocess_data,sample_wrong_answers, convert_to_int, convert_to_letter

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

load_embeddings = 0
if load_embeddings == 1 or 'word2index' not in vars():
    word2index, embedding_matrix = load_glove_embeddings('./embeddings/glove.6B.300d.txt', embedding_dim=300)

load_data = 0
if load_data == 1 or 'questions_intseq' not in vars():
    data = preprocess_data()
    
    # unpack data
    questions_intseq,answers_final_form,explain_intseq,lengths,cache = data
    maxlen_question,maxlen_explain,vocablen_question,vocablen_explain = lengths
    answers_intseq = cache['answers_intseq']
    wrong_answers = cache['wrong_answers']
    all_answer_options_intseq = cache['all_answer_options_intseq']
    answers = cache['answers']
    questions = cache['questions']
    
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    num_examples = questions_intseq.shape[0]


    # 50,20,30 split
    num_train = 1164
    num_test = 499
    shuffled_indices = np.arange(num_examples)
    np.random.shuffle(shuffled_indices)
    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train : num_train + num_test]
    
    
    
    
    
print("--- {:.2f} seconds ---".format(time.time() - start_time))

#%% keras model

NUM_HIDDEN_UNITS = 200

Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
Glove_embedding.trainable = False

input_explain = Input((maxlen_explain,) ,name = 'explanation')
X1 = Glove_embedding(input_explain)
X1 = Dropout(0.0)(X1)
exp_rep = Bidirectional(LSTM(NUM_HIDDEN_UNITS, name = 'explanation_representation', dropout = 0.5))(X1)

input_question = Input((maxlen_question,), name = 'question')

X2 = Glove_embedding(input_question)
X2 = Dropout(0.0)(X2)
question_rep = Bidirectional(LSTM(NUM_HIDDEN_UNITS, name = 'question_representation', dropout = 0.5))(X2)

rep_explain_ques = Add()([exp_rep,question_rep])

lstm_ans = Bidirectional(LSTM(NUM_HIDDEN_UNITS, name = 'answer_lstm', dropout = 0.5))

input_pos_ans = Input((23,))
input_neg_ans1 = Input((23,))
input_neg_ans2 = Input((23,))
input_neg_ans3 = Input((23,))

pos_ans = Glove_embedding(input_pos_ans)
neg_ans1 = Glove_embedding(input_neg_ans1)
neg_ans2 = Glove_embedding(input_neg_ans2)
neg_ans3 = Glove_embedding(input_neg_ans3)

pos_ans_rep = lstm_ans(pos_ans)
neg_ans_rep1 = lstm_ans(neg_ans1)
neg_ans_rep2 = lstm_ans(neg_ans2)
neg_ans_rep3 = lstm_ans(neg_ans3)

Cosine_similarity = Lambda(get_cosine_similarity,name = 'Cosine_similarity')

pos_similarity = Cosine_similarity([rep_explain_ques,pos_ans_rep])
neg_similarity1 = Cosine_similarity([rep_explain_ques,neg_ans_rep1])
neg_similarity2 = Cosine_similarity([rep_explain_ques,neg_ans_rep2])
neg_similarity3 = Cosine_similarity([rep_explain_ques,neg_ans_rep3])

loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])

prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])

#%% training

num_iter = 10
LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(LEARNING_RATE)

training_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1],outputs = loss)
training_model.compile(optimizer = OPTIMIZER,loss = _loss_tensor,metrics = [])
#print(model.summary())
dummy_labels = np.array([None]*num_train).reshape(num_train,1)

history_cache = dict()
val_loss = np.array([])
training_loss = np.array([])
for i in range(num_iter):
    print('running iteration {}...'.format(i))
    answers_intseq2 = sample_wrong_answers(wrong_answers)
    X_train = [explain_intseq[train_indices],questions_intseq[train_indices],answers_intseq[train_indices],answers_intseq2[train_indices]]
    history = training_model.fit(x = X_train,y = dummy_labels,batch_size = 128,validation_split = 0.2,epochs = 5)
    history_cache[i] = history.history
    val_loss = np.append(val_loss,history.history['val_loss'])
    training_loss = np.append(training_loss,history.history['loss'])
#val_loss = [history_cache[i]['val_loss'] for i in np.arange(num_iter)]
#val_loss = np.array(val_loss).reshape(-1,1)
#training_loss = [history_cache[i]['loss'] for i in np.arange(num_iter)]
#training_loss = np.array(training_loss).reshape(-1,1)

plt.plot(val_loss)
plt.plot(training_loss)

#%% predict
make_predictions = 1
if make_predictions == 1:
    prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = prediction)
    prediction_model.compile(optimizer = 'adam', loss = lambda y_true,y_pred: y_pred, metrics = [keras.metrics.categorical_accuracy])
    
    
    all_answer_options_intseq = np.array(all_answer_options_intseq)
    
    indices = test_indices # test_indices or training_indices
    temp1 = explain_intseq[indices]
    temp2 = questions_intseq[indices]
    temp3 = all_answer_options_intseq[indices,0,:]
    temp4 = all_answer_options_intseq[indices,1,:]
    temp5 = all_answer_options_intseq[indices,2,:]
    temp6 = all_answer_options_intseq[indices,3,:]  
    
    #model2.predict([temp1,temp2,temp3,temp4,temp5,temp6])
    
    predict_output = prediction_model.predict([temp1,temp2,temp3,temp4,temp5,temp6],batch_size = 1)
    predicted_ans = np.argmax(predict_output,axis = 1)
    print(predict_output)
    print(predicted_ans)
    
    
    int_ans = np.array([convert_to_int(letter) for letter,ans in answers])
    print(np.mean(predicted_ans == int_ans[indices]))
    
#    prediction_model.evaluate([temp1,temp2,temp3,temp4,temp5,temp6],correct_ans,verbose = False)

#%% save model
    
save_model = 0
if save_model == 1:
    model.save('./saved_models/rnn3.h5py')    
