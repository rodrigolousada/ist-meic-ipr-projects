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
from nltk.tokenize import word_tokenize
#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

########################
#      functions       #
########################

def fileSentences(filename):
    # print "entrou"
    with codecs.open(filename, "r", "latin-1") as file:
        lines = (file.read())#.split('\n')#.decode('utf-8')
        # print lines
    file.close()
    sentences = re.split(r'[\r\n]+',lines)
    return lines


def weightVect(sentences):
    vectorizer = TfidfVectorizer( use_idf=True )
    vecSpaceM = vectorizer.fit_transform(sentences)
    return vecSpaceM

# def calcSimilarity(listS,sizeDoc):
#
#     scores = {}
#     for i in xrange(sizeDoc):
#         soma = 0
#         for x in xrange(sizeDoc):
#             # print listS[i][x]
#             if listS[i][x] != 0:
#                 soma += listS[i][x]
#
#         scores.update({i:str(round(soma,7))})
#     return scores

def calcSim(vecSpaceModel,vecSpaceModelTrain,sizeDoc,num_words):

    scores = {}
    for i in xrange(sizeDoc):
        soma = 0
        listSimilarity = cosine_similarity(vecSpaceModel[i:i+1],vecSpaceModelTrain)
        for x in xrange(sizeDoc):
            # print listS[i][x]
            if (listSimilarity[0][x]) != 0:
                soma += listSimilarity[0][x]
        scores.update({i:str(round(soma,sizeDoc)/num_words)})
    print scores
    return scores

def bestSentences(dictSent,fileS,numb):
    sentSort = sorted(dictSent, key = dictSent.__getitem__,reverse=True) #,reverse=True
    #  sorted(dictSent.values(),reverse=True)
    bestS = []
    for i in xrange(numb):
        bestS.append(fileS[sentSort[i]])
    return bestS

########################
#         run          #
########################
if __name__ == '__main__':
    fpath = os.path.join("TeMario/Textos-fonte", "ce94jl10-a.txt")

    lines = fileSentences(fpath)
    # print fileS
    fileS = sent_tokenize(lines) #.decode('utf-16')
    words = word_tokenize(lines)
    num_words = len(words)
    sizeDoc = len(fileS) #[0:sizeDoc]

    vecSpaceModel = weightVect(fileS)
    #print vecSpaceModel

    # print listSimilarity
    score = calcSim(vecSpaceModel,vecSpaceModel,sizeDoc,num_words)
    # print score
    bestS = bestSentences(score,fileS,5)
    print bestS






#
