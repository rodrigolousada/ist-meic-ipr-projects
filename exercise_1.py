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
from sklearn.feature_extraction.text import _document_frequency
from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse as sp

################################################
#                   classes                    #
################################################

class TfidfTransformer_2(TfidfTransformer):
    def __init__(self):
        TfidfTransformer.__init__(self, use_idf = True)

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            idf = np.log10(float(n_samples) / df) +1 #remove 1? should I add TF?

            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features, format='csr')

            return self

class TfidfVectorizer_2(TfidfVectorizer):

    def __init__(self):
         TfidfVectorizer.__init__(self, use_idf = True)
         self._tfidf = TfidfTransformer_2()

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
    vectorizer = TfidfVectorizer_2()
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

def exercise_1_main(dir, file, nr):
    fpath = os.path.join(dir, file)

    lines = fileSentences(fpath)
    doc=lines.replace('\n', '')
    # fileS = sent_tokenize(lines)
    fileS = re.split(r'[\r\n]+',lines)

    matrixSimilarity = similarity([doc], fileS)
    #print matrixSimilarity
    scores = dictSimilarity(matrixSimilarity)
    #print scores

    bestS = bestSentences(scores,fileS,nr)
    printBestSent(bestS)

################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_1_main("TeMario/Textos-fonte", "ce94jl10-a.txt", 3)
