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


################################################
#                 functions                    #
################################################

def fileSentences(filename):
    # print "entrou"
    with codecs.open(filename, "r", "latin-1") as file:
        lines = (file.read())#.split('\n')#.decode('utf-8')
    file.close()
    return lines

def similarity(doc, fileSent):
    vectorizer = TfidfVectorizer( use_idf=True )
    vectorizer = vectorizer.fit(doc)

    vecSpaceM_sent = vectorizer.transform(fileSent)
    vecSpaceM_doc = vectorizer.transform(doc)

    listSimilarity = cosine_similarity(vecSpaceM_sent,vecSpaceM_doc)

    return listSimilarity

def dictSimilarity(listSimilarity):
    rangeMatrix = listSimilarity.shape[0]
    scores = {}

    for x in xrange(rangeMatrix):
        scores.update({x:str(round(listSimilarity[x][0],8))})
    return scores

def bestSentences(dictSent,fileS,numb):
    sentSort = sorted(dictSent, key = dictSent.__getitem__,reverse=True)
    bestS = []
    for i in xrange(numb):
        bestS.append(fileS[sentSort[i]])
    return bestS

def printBestSent(bestSent):
    for i in xrange(len(bestSent)):
        print bestSent[i] + "\n"



################################################
#                     run                      #
################################################

if __name__ == '__main__':
    fpath = os.path.join("TeMario/Textos-fonte", "ce94jl10-a.txt")

    lines = fileSentences(fpath)
    doc=lines.replace('\n', '')
    # fileS = sent_tokenize(lines)
    fileS = re.split(r'[\r\n]+',lines)

    matrixSimilarity = similarity([doc], fileS)

    scores = dictSimilarity(matrixSimilarity)

    bestS = bestSentences(scores,fileS,3)

    printBestSent(bestS)









    #
