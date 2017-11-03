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
from exercise_2 import *

################################################
#                 functions                    #
################################################

def calculateMMR(sentence, S, doc, lambda_symb):
    sum_sim_S = 0
    for sent in S:
        sum_sim_S += float(similarity([sent],[sentence]))
    document_similarity = similarity([doc],[sentence])
    mmr = float((1-lambda_symb) * document_similarity[0][0] - lambda_symb * sum_sim_S)
    return mmr


def exercise_4_main(nr, lambda_symb):
    statistics_1_list = []
    statistics_2_list = []
    for filename in os.listdir("TeMario/Textos-fonte")[1:]:
        print filename
        fpath = os.path.join("TeMario/Textos-fonte", filename)
        docEval = fileSentences(fpath)

        doc=(docEval.replace('\n', ' '))
        fileS = re.split(r'[\r\n\.]+',docEval.strip(" "))

        news_article_procedure = fileS[0:nr]
        #printBestSent(news_article_procedure)

        S = []
        not_in_S = []
        not_in_S += fileS[:-1]
        rangex = len(not_in_S)
        for i in xrange(5):
            mmr_list = []
            for sentence in fileS[:-1]:
                if len(sentence.split()) > 1:
                    mmr_list.append(calculateMMR(sentence, S, doc, lambda_symb))
                else:
                    mmr_list.append(0)
            index_to_S = mmr_list.index(max(mmr_list))
            S.append(not_in_S[index_to_S])
            del not_in_S[index_to_S]

        ideal_summary = getIdealSummary(filename)
        getStatistics(filename, statistics_1_list, ideal_summary, S)
        getStatistics(filename, statistics_2_list, ideal_summary, news_article_procedure)
        print"\n"

    print "\nEXERCISE MMR"
    print "MPrecision: " + str(getMPrecision(statistics_1_list))
    print "MRecall: " + str(getMRecall(statistics_1_list))
    print "MF1: " + str(getMF1(statistics_1_list))
    print "MAP: " + str(getMAP(statistics_1_list))

    print "\nEXERCISE NEW ARTICLE PROCEDURE"
    print "MPrecision: " + str(getMPrecision(statistics_2_list))
    print "MRecall: " + str(getMRecall(statistics_2_list))
    print "MF1: " + str(getMF1(statistics_2_list))
    print "MAP: " + str(getMAP(statistics_2_list))
################################################
#                     run                      #
################################################

if __name__ == '__main__':
    exercise_4_main(5, 0.8)
