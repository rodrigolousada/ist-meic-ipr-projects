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
from exercise_1 import  fileSentences, similarity, dictSimilarity, bestSentences, printBestSent, exercise_1_main

################################################
#                 functions                    #
################################################

def getFiles(path):
    allfiles = ""
    for filename in os.listdir(path)[1:]:
        fpath = os.path.join(path, filename)
        newfile = (fileSentences(fpath))
        # print (newfile).replace('\n', '')
        allfiles+=(newfile).replace('\n', '')
    return [allfiles]

def getIntersection(list1, list2):
    counter=0
    for elem in list1:
        if elem in list2:
            counter+=1
    return counter

def getPrecision(relevant_docs, answer_set):
    return getIntersection(relevant_docs, answer_set) / len(answer_set)

def getRecall(relevant_docs, answer_set):
    return getIntersection(relevant_docs, answer_set) / len(relevant_docs)

def getF1(precision, recall):
    return (2 * recall * precision) / (recall + precision)

def getPrecisionAt(nr, relevant_docs, answer_set):
    return getPrecision(relevant_docs, answer_set[0:nr])

def getAP(relevant_docs, answer_set):
    numerator = 0
    for i in xrange(len(relevant_docs)):
        if relevant_docs[i] in answer_set:
            numerator += getPrecisionAt(i, relevant_docs, answer_set)
    return numerator / len(relevant_docs)

def meanCalculator(statistics_list, nr):
    sum = 0
    for i in xrange(len(statistics_list)):
        sum += statistics_list[i][nr]
    return sum/len(statistics_list)

def getMPrecision(statistics_list):
    return meanCalculator(statistics_list,0)

def getMRecall(statistics_list):
    return meanCalculator(statistics_list,1)

def getMF1(statistics_list):
    return meanCalculator(statistics_list,2)

def getMAP(statistics_list):
    return meanCalculator(statistics_list,3)

def getIdealSummary(file):
    fpath_ideal = os.path.join("TeMario/ExtratosIdeais", "Ext-" + file)
    ideal_summary = fileSentences(fpath_ideal)
    return re.split(r'[\r\n]+',ideal_summary)

def getStatistics(file, i ,statistics_list, ideal_summary, bestS):
    precision = getPrecision(ideal_summary, bestS)
    recall = getRecall(ideal_summary, bestS)
    f1 = getF1(precision, recall)
    ap = getAP(ideal_summary, bestS)
    print "File: " + file
    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "F1: " + str(f1)
    print "AP: " + str(ap)
    return statistics_list.append([precision, recall, f1, ap])

def exercise_2_main(file):
    docs = getFiles("TeMario/Textos-fonte")
    statistics_list = []

    #for i add for every document
    fpath = os.path.join("TeMario/Textos-fonte", file)
    docEval = fileSentences(fpath)
    fileS = re.split(r'[\r\n]+',docEval)

    matrixSimilarity = similarity(docs, fileS)
    scores = dictSimilarity(matrixSimilarity)

    bestS = bestSentences(scores,fileS,5)
    printBestSent(bestS)

    for i in xrange(10):
        print "\n"

    ideal_summary = getIdealSummary(file)
    printBestSent(ideal_summary)

    getStatistics(file, 0, statistics_list, ideal_summary, bestS)


    #after having all files, calculate means

################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_2_main("ce94jl10-a.txt")
