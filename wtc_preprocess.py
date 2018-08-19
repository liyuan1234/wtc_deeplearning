#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:43:52 2018

@author: liyuan
"""

"""
Preprocess the textfile containing questions and explanations for elementary science questions
out 1664 questions and explanations
creates 2 files: explanations2.txt and questions2.txt
"""

file_dir1 = './wtc_data/questions.txt'
file_dir2 = './wtc_data/explanations_plaintext.nomercury.txt'






#%% read file dir 2
#file = open(file_dir2,encoding = 'utf-8')
#file.readlines(2)
#for i in range(100):
#    line = file.readline()
#    if 'Explanation' in line:
#        input()
#        while not 'Question' in line:
#            line = file.readline()
#        line = file.readline()
#    if 'Correct Answer' in line:
#        input()
#        print(line)
#    else:
#        print(line)
       
#%% make explanations_plaintext.nomercury(modified).txt --- remove extra line of questions: index
file2 = open('./wtc_data/explanations_starting.txt', 'w', encoding = 'utf-8')
with open('./wtc_data/explanations_plaintext.withmercury.txt',encoding = 'utf-8') as file:
    raw = file.readlines()
    for line in raw:
        if line.strip('Question: ').strip().isdigit():
            pass
        else:
            file2.write(line)
            
file2.close()


        
#%% make explanation_train.txt and question_train.txt

explanations_dir = './wtc_data/explanations2.txt'
questions_dir = './wtc_data/questions2.txt'
file = open('./wtc_data/explanations_starting.txt',encoding = 'utf-8')
explanations_out = open(explanations_dir,'w')
questions_out = open(questions_dir,'w')

expflag = False
questionsflag = False
explanations = ''
question = ''

raw = file.readlines()

counter = 0

for line in raw:
    '''if encounter question, then read lines until run into explanation. 
    Similarly if encounter explanation, read lines until next question'''
    counter = counter + 1
    
    if "Text of licensed questions can be requested at" in line:
        explanations_out.write(explanations+'\n')
        explanations = ''
        break
    
    if 'Question:' in line:
        questionsflag = True
        expflag = False
        if counter > 1:
            explanations_out.write(explanations+'\n')
            explanations = ''
    if 'Explanation: ' in line:
        questionsflag = False
        expflag = True
        
        question = question.replace('Question: ','')
        question = question.replace('Correct Answer',' ')
        question = question.replace('\n','')
        question = question.replace('\t','')
        question = question.replace('[0]:',' (A)')
        question = question.replace('[1]:',' (B)')
        question = question.replace('[2]:',' (C)')
        question = question.replace('[3]:',' (D)')
        
        
        question = question.replace(': 0',': A')
        question = question.replace(': 1',': B')
        question = question.replace(': 2',': C')
        question = question.replace(': 3',': D')
        questions_out.write(question + '\n')
        question = ''
    
    if questionsflag == True:
        question = question+line
    if expflag == True:
        if 'Explanation:' in line or line.isspace():
            pass
        else:
            explanations = explanations + line.split('(UID:')[0].replace('Explanation:','').strip()+'. '

explanations_out.write(explanations+'\n')
    
print('done!')
explanations_out.close()
questions_out.close()

file_exp = open(explanations_dir)
raw_exp = file_exp.readlines()

file_questions = open(questions_dir)
raw_questions = file_questions.readlines()
print(raw_exp[3])

            
    
