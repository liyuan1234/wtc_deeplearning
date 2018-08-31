#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:19:52 2018

@author: liyuan
"""

exp_tokenized = cache['exp_tokenized']
all_answer_options_intseq = np.array(all_answer_options_intseq)
prediction_model = Model(inputs = [input_explain,input_question,input_pos_ans,input_neg_ans1,input_neg_ans2,input_neg_ans3],outputs = prediction)
def convert_to_letter(index):
    if index == 0:
        return 'A'
    if index == 1:
        return 'B'
    if index == 2:
        return 'C'
    if index == 3:
        return 'D'


result = []
while True:
    test_index = int(input('input test index: '))
    i = test_index
    indices = test_indices[i:i+1]
    temp1 = explain_intseq[indices]
    temp2 = questions_intseq[indices]
    temp3 = all_answer_options_intseq[indices,0,:]
    temp4 = all_answer_options_intseq[indices,1,:]
    temp5 = all_answer_options_intseq[indices,2,:]
    temp6 = all_answer_options_intseq[indices,3,:]
    
    #question_index = int(input('input question index: '))
    predictions = prediction_model.predict([temp1,temp2,temp3,temp4,temp5,temp6])
    predicted_ans = np.argmax(predictions)
    correct_ans = answers[indices[0]][0]
    
    result.append(predicted_ans == convert_to_int(correct_ans))
    
    print(predictions)
    print(' '.join(questions[indices[0]]),'\n')
    print(' '.join(exp_tokenized[indices[0]]),'\n')
    print('predicted answer: {}'.format(convert_to_letter(predicted_ans)))
    print('correct answer: {}'.format(correct_ans))
    print('percent correct is : {}'.format(np.mean(result)))
    
    
    
def predict_answer(model2):
    model2.predict()
