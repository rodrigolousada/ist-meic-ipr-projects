########################
#      imports         #
########################
import nltk
import numpy as np
import re
import string
from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

########################
#      functions       #
########################

def fileSentences(filename):
    file = open(filename,"r")
    lines = file.read()
    file.close()
    sentences = sent_tokenize(lines)
    return sentences

def weightVect(sentences):
    vectorizer = TfidfVectorizer( use_idf=True )
    vecSpaceModel = vectorizer.fit_transform(sentences)
    #testvec = vectorizer.transform(test.data)
    return vecSpaceModel

def calcSimilarity(listSimilarity,sizeDoc):
    scores = {}
    for i in xrange(sizeDoc):
        soma = 0
        for x in xrange(sizeDoc):
            soma += listSimilarity[i][x]
        scores.update({i:str(round(soma,7))})
    return scores

def bestSentences(dictSent,fileSentences,numb):
    sentSort = sorted(dictSent, key = dictSent.__getitem__,reverse=True)
    #  sorted(dictSent.values(),reverse=True)
    bestSentences = []
    for i in xrange(numb):
        bestSentences.append(fileSentences[sentSort[i]])
    return bestSentences

########################
#         run          #
########################

fileSentences = fileSentences("test.txt")
print fileSentences
vecSpaceModel = weightVect(fileSentences)

sizeDoc = len(fileSentences)
listSimilarity = cosine_similarity(vecSpaceModel[0:sizeDoc],vecSpaceModel)

score = calcSimilarity(listSimilarity,sizeDoc)
bestSentences = bestSentences(score,fileSentences,3)
print bestSentences









#
