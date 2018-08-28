#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:19:52 2018

@author: liyuan
"""

while True:
    question_index = int(input('input question index: '))
    predictions = np.zeros([4,1])
    for i in range(4):
        model2 = Model(inputs = [input1,input2,input3],outputs = similarity1)
        prediction = model2.predict([explain_intseq[[question_index]],questions_intseq[[question_index]],all_answer_options_intseq[question_index][[i]]])
        predictions[i] = prediction
    
    print(predictions)
    print(' '.join(questions[question_index]))