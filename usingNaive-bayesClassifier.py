
# coding: utf-8

# In[7]:


import os
from collections import defaultdict
from pandas import read_table
import numpy as np
import math


# In[8]:


from ekonlpy.sentiment import MPCK
mpck = MPCK()


# In[19]:


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

#     load corpus data file
    def load_corpusData(self, path):
        corpusData = read_table(path, sep=',', header=None, names=None, encoding='utf-8')
        corpusData = np.array(corpusData)

        return corpusData
     
    
#     count words
    def count_words(self, training_set):
#         corpus data : dataDate, label
        counts = defaultdict(lambda : [0, 0])
        
#         read news file list
        news_list = os.listdir('data/news/')
#         read bonds file list
        bonds_list = os.listdir('data/bonds/')
        file_list = news_list + bonds_list
        
        for dataDate, label in training_set:            
            for file in file_list:
#                 find news and bonds
                if dataDate == file[5:15]:
#                     read news file
                    corpus = open('data/news/'+file, 'r', encoding='utf-8').read()
                    print("process news file name : ", file)
        
                    tokens = mpck.tokenize(corpus)
                    ngrams = mpck.ngramize(tokens)

#                     count feature
                    for ngram in ngrams+tokens:
                        counts[ngram][0 if label == 1 else 1] += 1
                    print("complete")
        
                elif dataDate == file[6:16]:
#                     read bonds file
                    corpus = open('data/bonds/'+file, 'r', encoding='utf-8').read()
                    print("process bonds file name : ", file)

#                     make corpus like nouns(NNG), adjectives (VA, VAX), adverbs (MAG), verbs (VA) and negations using eKoNLPy
                    tokens = mpck.tokenize(corpus)
                    ngrams = mpck.ngramize(tokens)
        
#                     count feature
                    for ngram in ngrams+tokens:
                        counts[ngram][0 if label == 1 else 1] += 1
                    print("complete")
        
        return counts

    def word_probabilities(self, counts, total_class0, total_class1, k):
        # get word, p(w|positive), p(w|negative)
        return [(w, (class0 + k) / (total_class0 + 2*k), (class1 + k) / (total_class1 + 2*k))
                for w, (class0, class1) in counts.items()]        

    def train(self, trainfile_path):
        training_set = self.load_corpusData(trainfile_path)

#         get prior
        positive = len([1 for _, label in training_set if label == 1])
        negative = len(training_set) - positive

#         train
        word_counts = self.count_words(training_set)
        
        self.word_probs = self.word_probabilities(word_counts, positive, negative, self.k)

#         save words
        for noOfWord in range(len(self.word_probs)):
            if self.word_probs[noOfWord][1]/self.word_probs[noOfWord][2] > 1:
                with open('data/res/positive.csv', 'a', encoding='utf-8') as f:
                    f.write(self.word_probs[noOfWord][0]+'\n')
            else:
                with open('data/res/negative.csv', 'a', encoding='utf-8') as f:
                    f.write(self.word_probs[noOfWord][0]+'\n') 


# In[20]:


model = NaiveBayesClassifier()
# save result file in res folder
model.train(trainfile_path='data/labeledCallRate.csv')

