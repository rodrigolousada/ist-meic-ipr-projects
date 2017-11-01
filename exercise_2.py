########################
#      imports         #
########################
import codecs
import glob, os
import nltk
import numpy as np
import re
import sys
import string
from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from exercise_1 import  fileSentences, weightVect, calcSimilarity, bestSentences

########################
#      functions       #
########################

def getFiles(path):
    allfiles = []
    for filename in os.listdir(path)[1:]:
        fpath = os.path.join(path, filename)
        newfile = fileSentences(fpath)
        allfiles+=(newfile)
    return allfiles


########################
#         run          #
########################
if __name__ == '__main__':

    fpath = os.path.join("TeMario/Textos-fonte", "ce94jl10-a.txt")
    fileS = fileSentences(fpath)
    # print fileS
    sizeDoc = len(fileS)
    # vecSpaceModel = weightVect(fileS)

    allDoc = getFiles("TeMario/Textos-fonte")
    #print allDoc
    # vecSpaceModelAllFiles = weightVect(allDoc)
    # print vecSpaceModelAllFiles.shape[1]

    # listSimilarity = cosine_similarity(vecSpaceModel,vecSpaceModelAllFiles)
#############################

    vectorizer = TfidfVectorizer( use_idf=True )
    vecSpacefit = vectorizer.fit(allDoc)
    doc = vectorizer.transform(fileS)
    print doc.shape[1]
    docs = vectorizer.transform(allDoc)
    print docs.shape[1]

    listSimilarity = cosine_similarity(doc,docs)
    print len(listSimilarity)
    print sizeDoc
    # print listSimilarity

    score = calcSimilarity(listSimilarity,sizeDoc)
    bestS = bestSentences(score,fileS,5)
    print bestS










#
