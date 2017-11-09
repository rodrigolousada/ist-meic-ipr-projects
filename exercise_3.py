################################################
#                   imports                    #
################################################
#import nltk
#nltk.download('floresta')

import codecs
import glob, os
import nltk
import numpy as np
import re
import sys
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from exercise_1 import  fileSentences, similarity, dictSimilarity, bestSentences, printBestSent, exercise_1_main
from exercise_2 import  getFiles, getIntersection, getPrecision, getRecall, getF1, getAP, meanCalculator, getMPrecision, getMRecall, getMF1, getMAP, getIdealSummary,  getStatistics, exercise_2_main
from nltk.corpus import floresta
from nltk import ngrams
import scipy.sparse as sp

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize

stopwords = nltk.corpus.stopwords.words('portuguese')

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

            nonzero_array = np.nonzero(X)
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

            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

class BM25Vectorizer(TfidfVectorizer):

    def __init__(self):
         TfidfVectorizer.__init__(self, use_idf = True, smooth_idf=False, ngram_range=(2,2), stop_words=stopwords)
         self._tfidf = BM25Transformer()


################################################
#                 functions                    #
################################################
def similarityBiGram(doc, fileSent):
    vectorizer = BM25Vectorizer()

    docsWords = word_tokenize(doc[0])
    vectorizer = vectorizer.fit(docsWords)

    vecSpaceM_sent = vectorizer.transform(fileSent)
    vecSpaceM_doc = vectorizer.transform(docsWords)

    listSimilarity = cosine_similarity(vecSpaceM_sent,vecSpaceM_doc)
    return listSimilarity




def docClean(document):
    newdoc = ""
    newdoc += ([i for i in document.split() if i not in stopwords])
    return document

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t

def tokenize():
    tsents = floresta.tagged_sents()
    tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
    train = tsents[100:]
    test = tsents[:100]
    tagger0 = nltk.DefaultTagger('n')
    tagger1 = nltk.UnigramTagger(train, backoff=tagger0)
    return tagger1


def tagger(doc,tag1):
    tags = {}
    for sentence in doc:
        tags.update(tag1.tag(word_tokenize(sentence)))
    return tags


def exercise_3_main(file):
    docs = getFiles("TeMario/Textos-fonte")

    statistics_3_list = []
    tagForWords = tokenize()

    for filename in os.listdir("TeMario/Textos-fonte")[1:]:
        fpath = os.path.join("TeMario/Textos-fonte", filename)
        docEval = fileSentences(fpath)
        fileS = re.split(r'[\r\n\.]+',docEval.strip(" "))

        tokenizeDoc = tagger(fileS,tagForWords)

        matrixSimilarity = similarityBiGram(docs, tokenizeDoc)

        scores = dictSimilarity(matrixSimilarity)

        bestS = bestSentences(scores,fileS,5)


        ideal_summary = getIdealSummary(filename)
        # printBestSent(ideal_summary)

        getStatistics(filename, statistics_3_list, ideal_summary, bestS)

    exercise_2_main(filename)
    #after having all files, calculate means
    print "\nEXERCISE 3"
    print "MPrecision: " + str(getMPrecision(statistics_3_list))
    print "MRecall: " + str(getMRecall(statistics_3_list))
    print "MF1: " + str(getMF1(statistics_3_list))
    print "MAP: " + str(getMAP(statistics_3_list))

################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_3_main("ce94jl10-a.txt")
