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
        newfile = fileSentences(fpath)
        # print (newfile).replace('\n', '')
        allfiles+=((newfile).replace('\n', ' ')).strip()
    return [allfiles]

def getIntersection(list1, list2):
    counter=0

    for elem in list1:
        if elem in list2:
            counter+=1
    return counter

def getPrecision(relevant_docs, answer_set):
    return float(getIntersection(relevant_docs, answer_set) / float(len(answer_set)))

def getRecall(relevant_docs, answer_set):
    return float(getIntersection(relevant_docs, answer_set) / float(len(relevant_docs)))

def getF1(precision, recall):
    return float((2 * recall * precision) / float((recall + precision)))


def getAP(relevant_docs, answer_set):
    numerator = 0
    for i in xrange(len(relevant_docs)):
        if relevant_docs[i] in answer_set:
            index = answer_set.index(relevant_docs[i])
            numerator += getPrecision(relevant_docs[0:i+1], answer_set[0:index+1])
    return float(numerator / float(len(relevant_docs)))

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
    return re.split(r'[\r\n\.]+',ideal_summary.strip(" "))

def getStatistics(file, statistics_list, ideal_summary, bestS):
    ideal_summary = [x.strip(' ') for x in ideal_summary]
    bestS = [x.strip(' ') for x in bestS]

    precision = getPrecision(ideal_summary, bestS)

    recall = getRecall(ideal_summary, bestS)

    if precision + recall != 0:
        f1 = getF1(precision, recall)
        ap = getAP(ideal_summary, bestS)
    else:
        f1 = 0
        ap = 0
    print "File: " + file
    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "F1: " + str(f1)
    print "AP: " + str(ap)
    return statistics_list.append([precision, recall, f1, ap])

def exercise_2_main(file):
    docs = getFiles("TeMario/Textos-fonte")

    statistics_list = []

    for filename in os.listdir("TeMario/Textos-fonte")[1:]:
        fpath = os.path.join("TeMario/Textos-fonte", filename)
        docEval = fileSentences(fpath)

        fileS = re.split(r'[\r\n\.]+',docEval.strip(" "))

        matrixSimilarity = similarity(docs, fileS)

        scores = dictSimilarity(matrixSimilarity)

        bestS = bestSentences(scores,fileS,5)
        printBestSent(bestS)

        for i in xrange(3):
             print "\n"
        ideal_summary = getIdealSummary(filename)
        printBestSent(ideal_summary)

        getStatistics(filename, statistics_list, ideal_summary, bestS)
    #after having all files, calculate means

    print "MPrecision: " + str(getMPrecision(statistics_list))
    print "MRecall: " + str(getMRecall(statistics_list))
    print "MF1: " + str(getMF1(statistics_list))
    print "MAP: " + str(getMAP(statistics_list))

################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_2_main("ce94jl10-a.txt")
