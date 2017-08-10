''' file:           co-occurence-matrix.py
    author:         vpawar@cs.umass.edu
    description:    A simple word embeddings generating using a co-occurence
                    matrix followd by SVD
'''


import numpy as np
from scipy.sparse import dok_matrix
import nltk
from nltk.stem import WordNetLemmatizer as lemm
lemm = lemm().lemmatize 
import matplotlib as plt
import pickle
import time

window_size = 3


def build_matrix(words,window_size):

    print(build_matrix.__name__)

    num_words = len(words)

    # M = dok_matrix((num_words,num_words),dtype=int)
    M = np.zeros([num_words,num_words])

    for i in range(num_words):
        # print(words[i])
        for j in range(i-5,i+5):
            if j >= 0 and j < num_words:
                M[i,j]+=1
    return M
        
        
def train():

    with open("data/dickens-a_tale_of_two_cities.txt","r") as f:
        txt = f.read()
        words = nltk.tokenize.word_tokenize(txt)
        words = [ lemm(w.lower()) for w in words]
        print(len(words))
        words = list(set(words))
        print(len(words))
        words_dict = { word:i for i,word in enumerate(words) }

        M  = build_matrix(words,window_size)
        print(np.linalg.svd.__name__)
        start = time.time()
        U, s, Vh = np.linalg.svd(M,full_matrices=False)
        # print("time taken = ",time.time()-start,sep=" ")

        print("storing embeddings to file")
        pickle.dump(U,open("embeddings.pickle","w"))
        pickle.dump(words_dict,open("words_dict.pickle","w"))


def test():
    
        U = pickle.load(open("embeddings.pickle","r"))
        words_dict = pickle.load(open("words_dict.pickle","r"))

        print(np.dot(U[words_dict["dog"]],U[words_dict["cat"]]))
        print(np.dot(U[words_dict["pet"]],U[words_dict["cat"]]))
        print(np.dot(U[words_dict["chair"]],U[words_dict["furniture"]]))
        print(np.dot(U[words_dict["furniture"]],U[words_dict["furniture"]]))

if __name__=="__main__":
    test()
    # train()
    


