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
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

        fileSent = tuple(re.split(r'[\r\n\.]+',lines.strip(" ")))
        data.append(fileSent[:-1])
    print(data)
    return data

def fileRead(filename):
	with codecs.open(filename, "r", "latin-1") as file:
		lines = (file.read())#.split('\n')#.decode('utf-8')
	file.close()
	return lines.lower()

def trainData(data):
    count_vect = CountVectorizer()
    X_train_counts = (count_vect.fit_transform(data))
    tfidf_transformer = TfidfTransformer_2()
    tfidf_transformer = (tfidf_transformer.fit(X_train_counts))
    X_train_tfidf = (tfidf_transformer.transform(X_train_counts))
    print(X_train_tfidf)
    return X_train_tfidf

def naive_bayes(train,target, test):
    clf = MultinomialNB().fit(train, target)
    clf = clf.fit(train, target)
    predicted = clf.predict(test)
    print(predicted)
    # return predicted

def appendTrainTarget(train, target):
    result = []
    for x in range(len(train)):
        newres = (train[x],target[x])
        result.append(newres)
    return result

def exercise_3_main():
    train_data = getDataPath("TeMário_2006/Originais")
    train_target = getDataPath("TeMário_2006/SumáriosExtractivos")
    test_data = getDataFromDir("TeMario/Textos-fonte")
    test_target = getDataFromDir("TeMario/ExtratosIdeais")
    train = trainData(train_data)
    naive_bayes(train, train_target, test_data)
    # tfidf_transformer = TfidfVectorizer_2()
    # tfidf_transformer = (tfidf_transformer.fit(train_data))
    # new_train = tfidf_transformer.transform(train_data)


    # from sklearn import preprocessing
    # enc = preprocessing.LabelEncoder()
    # enc= enc.fit(train_data)
    # train = enc.transform(train_data)
    # target = enc.transform(train_target)
    #
    # clf = MultinomialNB().fit(train, target)
    # predicted = clf.predict(test_data)
    # print("--------------------------------------")
    # print(predicted)

    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer_2()), ('clf', MultinomialNB()),])
    # text_clf = text_clf.fit(train_data, train_target)
    # predicted = text_clf.predict(test_data)

    # for x in range(100):
    #     print("----New-----")
    #
    #     print(predicted[x])
    #     print("------")
    #     print(test_target[x])
    #
    #     print(predicted[x]==test_target[x])
    # trained_data = trainData(train_data)
    # predict = naive_bayes(trained_data,train_target,test_data)
    # doc=(lines.replace('\n', ' '))
    # fileSent = re.split(r'[\r\n\.]+',lines.strip(" "))


################################################
#                     run                      #
################################################

if __name__ == '__main__':
	mainS = exercise_3_main()
