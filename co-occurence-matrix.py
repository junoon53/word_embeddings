''' file:           co-occurence-matrix.py
    author:         vpawar@cs.umass.edu
    description:    A simple word embeddings generating using a co-occurence
                    matrix followd by SVD
'''


import numpy as np
import nltk
import matplotlib as plt

window_size = 3


def build_matrix(words,window_size):

    num_words = len(words)

    M = np.zeros([num_words,num_words])

    for i in range(words):
        for j in range(i-5,i+5):
            M[i,j]+=1
        
        

with open("data/dickens-a_tale_of_two_cities.txt","r") as f:
    txt = f.read()
    words = nltk.tokenize.word_tokenize(txt)
    M  = build_matrix(words,window_size)
    U, s, Vh = np.linalg.svd(M,full_matrices=False)

    for i in xrange(len(words)):
        plt.text(U[i,0], U[i,1], words[i])
    


