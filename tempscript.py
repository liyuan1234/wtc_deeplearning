#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:48:10 2018

@author: liyuan
"""
#
file = open('./wtc_data/explanations2.txt','r')
raw_exp = file.readlines()
exp_tokenized = []
for paragraph in raw_exp:
    tokenized_paragraph = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(paragraph)]
    exp_tokenized.append(tokenized_paragraph)

#for x in exp_tokenized:
    
    
sent_lengths = [[len(x) for x in para] for para in exp_tokenized]
max_sent_length = max([max(x) for x in sent_lengths])
max_sent = max([len(x) for x in sent_lengths])

exp_intseq = []
for i in range(len(exp_tokenized)):
    temp = [tokenized_sentence_to_intseq(sentence,word2index) for sentence in exp_tokenized[i]]
    temp = pad_sequences(temp,maxlen = max_sent_length,value = 400000)
    filler = np.ones([max_sent,max_sent_length])*400000
    exp_intseq.append(np.vstack([temp,filler])[0:max_sent])

exp_intseq = np.array(exp_intseq)



def convert_to_intseq_and_pad(exp_tokenized,max_sent,max_sent_length,word2index):
    """
    e.g. [['I','like','eggs'],['I','also','like','bacon']] 
    to 
    [400000,400000,    41,   117,  5130]
    [400000,41    ,    52,   117, 10111]
    [400000,400000,400000,400000,400000]
    ...
    [400000,400000,400000,400000,400000]    
    """
    exp_intseq = []
    for i in range(len(exp_tokenized)):
        temp = [tokenized_sentence_to_intseq(sentence,word2index) for sentence in exp_tokenized[i]]
        temp = pad_sequences(temp,maxlen = max_sent_length,value = 400000)
        filler = np.ones([max_sent,max_sent_length])*400000
        exp_intseq.append(np.vstack([temp,filler])[0:max_sent])
    exp_intseq = np.array(exp_intseq)
    return exp_intseq






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






#exp_intseq = [tokenized_sentence_to_intseq(sentence,word2index) for sentence in exp_tokenized]

#%%


max_sentences = 27
max_sentence_length = 86

num_hidden_units = hyperparameters['num_hidden_units']
dropout_rate     = hyperparameters['dropout_rate']
learning_rate    = hyperparameters['learning_rate']
optimizer        = hyperparameters['optimizer']

RNN = Bidirectional(GRU(num_hidden_units, name = 'answer_lstm', dropout = dropout_rate,return_sequences = False))
Glove_embedding = Embedding(input_dim = len(word2index),output_dim = 300, weights = [embedding_matrix], name = 'glove_embedding')
Glove_embedding.trainable = False
Reshaper = Reshape([86])



exp_representations = []

input_explain = Input(shape = [27,86])
input_question = Input([maxlen_question])


exp_split = Lambda(lambda x: tf.split(x,max_sentences,1))(input_explain)
for exp in exp_split:
    exp = Reshaper(exp)
    exp = Glove_embedding(exp)
    exp = RNN(exp)
    exp_representations.append(exp)
exp_representation = Average()(exp_representations)

question_representation = Glove_embedding(input_question)
question_representation = RNN(question_representation)

combined_rep = Add()([exp_representation,question_representation])

input_pos_ans = Input((23,))
input_neg_ans1 = Input((23,))
input_neg_ans2 = Input((23,))
input_neg_ans3 = Input((23,))

pos_ans = Glove_embedding(input_pos_ans)
neg_ans1 = Glove_embedding(input_neg_ans1)
neg_ans2 = Glove_embedding(input_neg_ans2)
neg_ans3 = Glove_embedding(input_neg_ans3)

pos_ans_rep  = RNN(pos_ans)
neg_ans_rep1  = RNN(neg_ans1)
neg_ans_rep2  = RNN(neg_ans2)
neg_ans_rep3  = RNN(neg_ans3)

Cosine_similarity = Lambda(get_cos_similarity ,name = 'Cosine_similarity')

pos_similarity  = Cosine_similarity([combined_rep,pos_ans_rep])
neg_similarity1 = Cosine_similarity([combined_rep,neg_ans_rep1])
neg_similarity2 = Cosine_similarity([combined_rep,neg_ans_rep2])
neg_similarity3 = Cosine_similarity([combined_rep,neg_ans_rep3])

loss = Lambda(hinge_loss, name = 'loss')([pos_similarity,neg_similarity1])
#loss = Lambda(lambda x: K.tf.losses.hinge_loss(x[0],x[1],weights = 3), name = 'loss')([pos_similarity,neg_similarity1])

prediction = Concatenate(axis = -1, name = 'prediction')([pos_similarity,neg_similarity1,neg_similarity2,neg_similarity3])
prediction = Activation('softmax')(prediction)
