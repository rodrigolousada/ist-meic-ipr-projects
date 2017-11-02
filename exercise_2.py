################################################
#                   imports                    #
################################################
import codecs
import glob, os
import nltk
import numpy as np
import re
import sys
import string
#from nltk.tokenize import sent_tokenize, word_tokenize
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from exercise_1 import  fileSentences, similarity, dictSimilarity, bestSentences, printBestSent

################################################
#                 functions                    #
################################################

def getFiles(path):
    allfiles = ""
    for filename in os.listdir(path)[1:]:
        fpath = os.path.join(path, filename)
        newfile = (fileSentences(fpath))#
        # print (newfile).replace('\n', '')
        allfiles+=(newfile).replace('\n', '')
    return [allfiles]



################################################
#                     run                      #
################################################

if __name__ == '__main__':

    fpath = os.path.join("TeMario/Textos-fonte", "ce94jl10-a.txt")
    docs = getFiles("TeMario/Textos-fonte")

    docEval = fileSentences(fpath)
    fileS = re.split(r'[\r\n]+',docEval)
    
    matrixSimilarity = similarity(docs, fileS)
    scores = dictSimilarity(matrixSimilarity)

    bestS = bestSentences(scores,fileS,5)

    printBestSent(bestS)














#
