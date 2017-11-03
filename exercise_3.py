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

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
################################################
#                   classes                    #
################################################

class BM25Transformer(TfidfTransformer):

    def __init__(self):
        TfidfTransformer.__init__(self, use_idf = True, smooth_idf=False)
        self._avgdl=0

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

            idf = np.log10((float(n_samples) - df + 0.5) / (df+0.5))
            self._avgdl = avgdl = np.average(X.sum(axis=1))
            #print self._avgdl
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features, format='csr')

            return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))

            k1 = 1.2
            b = 0.75

            print X
            nonzero_array = np.nonzero(X)
            print nonzero_array
            lines = nonzero_array[0:1][0]
            columns = nonzero_array[1:2][0]

            document_sum = X.sum(1)
            for i in xrange(len(lines)):
                document = lines[i]
                term = columns[i]

                document_lenght = document_sum[document,0]
                term_frequency = X[document,term]
                score=(term_frequency*(k1+1))/(term_frequency+k1*(1-b+b*(document_lenght/self._avgdl)))
                X[document,term]=score

            print X

            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

class BM25Vectorizer(TfidfVectorizer):

    def __init__(self):
         TfidfVectorizer.__init__(self, use_idf = True, smooth_idf=False)
         self._tfidf = BM25Transformer()

################################################
#                 functions                    #
################################################

def fileSentences(filename):
    with codecs.open(filename, "r", "latin-1") as file:
        lines = (file.read())#.split('\n')#.decode('utf-8')
    file.close()
    return lines

def similarity(doc, fileSent):
    vectorizer = BM25Vectorizer()
    vectorizer = vectorizer.fit(fileSent)

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
        bestS.append((fileS[sentSort[i]]))
        #re.sub(r'^\s|\s$','',
    return bestS

def printBestSent(bestSent):
    for i in xrange(len(bestSent)):
        print bestSent[i] + "\n"

def exercise_1_main(dir, file, nr):
    fpath = os.path.join(dir, file)

    lines = fileSentences(fpath)
    doc=(lines.replace('\n', ' '))
    # fileS = sent_tokenize(lines)
    fileS = re.split(r'[\r\n\.]+',lines.strip(" "))

    matrixSimilarity = similarity([doc], fileS)
    #print matrixSimilarity
    scores = dictSimilarity(matrixSimilarity)
    #print scores

    bestS = bestSentences(scores,fileS,nr)
    printBestSent(bestS)
    return bestS
################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_1_main("TeMario/Textos-fonte", "ce94jl10-a.txt", 3)
