
# coding: utf-8

# In[26]:


# -*- encoding: utf-8 -*-


# In[2]:


import os
import pandas as pd
from collections import defaultdict
import csv


# In[3]:


from ekonlpy.sentiment import MPCK
mpck = MPCK()


# In[4]:


file_list = os.listdir('data/minutes/txt/')


# In[30]:


NoOfPositiveNgrams, NoOfnegativeNgrams = 0, 0
for file in file_list:
    
    positiveNgram = pd.read_csv('data/res/positive.csv', sep='\n', header=None, names=['positiveNgram'], encoding='utf-8')
    negativeNgram = pd.read_csv('data/res/negative.csv', sep='\n', header=None, names=['negativeNgram'], encoding='utf-8')

#     read minutes file    
    minutes = open('data/minutes/txt/'+file, 'r', encoding='utf-8').read()   
    
#     make ngrams
    minutesTokens = mpck.tokenize(minutes)
    minutesNgrams = mpck.ngramize(minutesTokens)
    
    for mN, pN, nN in zip(minutesNgrams+minutesTokens, positiveNgram.positiveNgram, negativeNgram.negativeNgram):
        if mN == pN:
            NoOfPositiveNgrams = NoOfPositiveNgrams + 1            
            print('Positive match ngrams: ')
            print(mN, '==', pN)
        elif mN == nN:
            NoOfnegativeNgrams = NoOfnegativeNgrams + 1
            print('negative match ngrams: ')
            print(mN, '==', nN)
        
print('최종값 : ')
print('NoOfPositiveNgrams : ', NoOfPositiveNgrams)
print('NoOfnegativeNgrams : ', NoOfnegativeNgrams)

# get polarity score of sentence
polarityScore_sentence = (NoOfPositiveNgrams - NoOfnegativeNgrams)/(NoOfPositiveNgrams + NoOfnegativeNgrams)

# save file
with open('data/res/polarityScore.csv', 'a', encoding='utf-8') as f:
    f.write(file[3:11]+","+str(polarityScore_sentence)+'\n')

