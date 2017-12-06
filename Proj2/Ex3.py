################################################
#                    Ex3					   #
#               Project by:                    #
#		Group 13							   #
#		Sofia Aparicio 81105				   #
#		Rodrigo Lousada 81115				   #
#		Rogerio Cabaco 81470				   #
################################################

################################################
#                   imports                    #
################################################
from Ex2 import TfidfTransformer_2, TfidfVectorizer_2, Graph, Vertex, Edge
import re, pdb, sys, math, nltk, glob, os, codecs, string
import scipy.sparse as sp
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer,CountVectorizer,_document_frequency
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize

punctuation = ["!","#","\''","*",",",":",";","<","=",">","@","[","]","^","_","{","|","}","\""]
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
            for i in range(len(lines)):
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
         TfidfVectorizer.__init__(self, use_idf = True, smooth_idf=False) #, stop_words=stopwords
         self._tfidf = BM25Transformer()

################################################
#                 functions                    #
################################################
def getDataPath(path):
    for pathPlus in os.listdir(path)[1:]:
        fileDir = os.path.join(path, pathPlus)
        dataFromDir = getDataFromDir(fileDir)
    return dataFromDir

def getDataFromDir(path):
    data = []
    for filename in os.listdir(path)[1:]:
        fileName = os.path.join(path, filename)
        lines = fileRead(fileName)
        # fileSent = re.split(r'[\r\n\.]+',lines.strip(" "))
        # sentences = []
        # paragraphs = [p for p in lines.split('\n') if p]
        # for paragraph in paragraphs:
        #     sentences += sent_tokenize(paragraph)

        sentences = []
        sentences_final = []
        paragraphs = [p for p in lines.split('\n') if p]
        for paragraph in paragraphs:
            sentences += sent_tokenize(paragraph)
        for sentence in sentences:
            if sentence.strip(" ") != "(...)":
                sentences_final.append(sentence.strip(" "))
        data.append(sentences_final)


    return data

def fileRead(filename):
	with codecs.open(filename, "r", "latin-1") as file:
		lines = (file.read())#.split('\n')#.decode('utf-8')
	file.close()
	return lines.lower()

def trainData(train, target):
    matrixTrain = []
    labels = []
    for x in range(len(train)):
        calcs = getCalcs(train[x], target[x])
        labels += calcs[1]
        matrixTrain += calcs[0]
    return (matrixTrain,labels)

def getCalcs(doc,summary):
    matrixDoc = []
    labels = []
    for index in range(len(doc)):
        # if len(doc[index]) > 1:
        labels.append(1 if doc[index] in summary else 0)
        matrixDoc.append([index+1 , tfidfSim(doc,doc[index]) ]) #, bm25Sim(doc,doc[index]),
    return(matrixDoc,labels)

def tfidfSim(doc, fileSent):
    vectorizer = TfidfVectorizer_2()
    sentWords = word_tokenize("".join((char for char in fileSent if char not in string.punctuation)))
    vectorizer = vectorizer.fit(sentWords)
    vecSpaceM_sent = vectorizer.transform(sentWords)
    vecSpaceM_doc = vectorizer.transform(doc)
    listSimilarity = cosine_similarity(vecSpaceM_sent,vecSpaceM_doc)
    return (listSimilarity).sum()

def bm25Sim(doc, fileSent):
    sentWords = word_tokenize("".join((char for char in fileSent if char not in string.punctuation)))
    vectorizer = BM25Vectorizer()
    vectorizer = vectorizer.fit(sentWords)

    vecSpaceM_sent = vectorizer.transform(doc)
    vecSpaceM_doc = vectorizer.transform(sentWords)

    listSimilarity = cosine_similarity(vecSpaceM_sent,vecSpaceM_doc)

    return listSimilarity

def perceptron(train, labels, test):
    ppt = Perceptron(max_iter = 300, eta0 = 0.1, random_state = 0)
    print(train)

    ppt.fit(train, labels)
    predict = ppt.predict(test)
    return predict

def exercise_3_main():
    train_data = getDataPath("TeMário_2006/Originais")
    train_target = getDataPath("TeMário_2006/SumáriosExtractivos")
    test_data = getDataFromDir("TeMario/Textos-fonte")
    test_target = getDataFromDir("TeMario/ExtratosIdeais")

    train = trainData(train_data,train_target) #train[0] matrix train[1] labels
    test = getCalcs(test_data[3],test_target[3])

    result = perceptron(train[0],train[1],test[0])

    print(test[0])
    print(len(test[0]))
    print(result)
    for index in range(len(result)):

        if result[index] !=0:
            print(index, " - ",test_data[3][index])

    # naive_bayes(train, train_target, test_data)

################################################
#                     run                      #
################################################

if __name__ == '__main__':
	mainS = exercise_3_main()
