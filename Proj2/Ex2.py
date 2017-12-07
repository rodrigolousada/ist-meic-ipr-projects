################################################
#                    Ex2					   #
#               Project by:                    #
#		Group 13							   #
#		Sofia Aparicio 81105				   #
#		Rodrigo Lousada 81115				   #
#		Rogerio Cabaco 						   #
################################################

################################################
#                   imports                    #
################################################
from Ex1 import fileRead, exercise_1_main
import re, pdb, sys, math, nltk, glob, os, codecs, string
import scipy.sparse as sp
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, _document_frequency
from sklearn.metrics.pairwise import cosine_similarity

################################################
#                 constants                    #
################################################
THRESHOLD = 0.2
RESID_PROB = 0.15 #(d)
MAXITERATIONS = 50
SENT_SUM = 5

################################################
#                   classes                    #
################################################
class TfidfTransformer_2(TfidfTransformer):
	def __init__(self):
		TfidfTransformer.__init__(self, use_idf = True, smooth_idf=False)

	def fit(self, X, y=None):
		"""Learn the idf vector (global term weights)
		Parameters
		X : sparse matrix, [n_samples, n_features]
			a matrix of term/token counts
		"""
		if not sp.issparse(X):
			X = sp.csc_matrix(X)
		if self.use_idf:
			n_samples, n_features = X.shape
			df = _document_frequency(X)
			idf = np.log10(float(n_samples) / df) #remove 1? should I add TF?
			self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features, format='csr')

			return self

class TfidfVectorizer_2(TfidfVectorizer):

	def __init__(self):
		TfidfVectorizer.__init__(self, use_idf = True, smooth_idf=False)
		self._tfidf = TfidfTransformer_2()


class Graph:
	def __init__(self, listVertices):
		self.Vertices = self.createAllVert(listVertices)
		self.Edges = self.createAllEdges()
		self.setVerticesPriors()
		self.setEdgeWeight()

	def printGraph(self):
		for edge in self.Edges:
			print(edge.Vertex1.Sentence, ' /n ', edge.Vertex2.Sentence)
			print(edge.Vertex1.PriorWeight,' /n ', edge.Vertex2.PriorWeight )
			print(edge.Weight)

	def setVerticesPriors(self):
		numVertices = self.numbVertices()
		for vertex in self.Vertices:
			priorWeight = self.priorCalc(numVertices, self.Vertices.index(vertex))
			vertex.setPriorWeight(priorWeight)

	def priorCalc(self, numVertices, index):
		index += 1
		first = 1/index
		second = 1/(numVertices - index + 1)
		return max(first, second)

	def setEdgeWeight(self):
		for edge in self.Edges:
			edgeWeight = self.edgeWeightCalc(edge)
			edge.setWeight(edgeWeight)

	def edgeWeightCalc(self, edge):
		vectorizer = TfidfVectorizer_2()
		sent1_words = word_tokenize(edge.Vertex1.Sentence)
		sent2_words = word_tokenize(edge.Vertex2.Sentence)

		vectorizer = vectorizer.fit(sent1_words)

		vecSpaceM_sent1 = vectorizer.transform(sent1_words)
		vecSpaceM_sent2 = vectorizer.transform(sent2_words)

		listSimilarity = cosine_similarity(vecSpaceM_sent1,vecSpaceM_sent2)
		return (listSimilarity).sum()

	def createAllVert(self, listVertices):
		vertList = []
		for sent in listVertices:
			if len(sent) > 1:
				newVertex = Vertex(sent)
				vertList.append(newVertex)
		return vertList

	def createAllEdges(self):
		edgeList = []
		lenList = len(self.Vertices)
		for index in range(lenList):
			for index2 in range(index+1, lenList):
				cosSim = self.similarity((self.Vertices[index]).Sentence,(self.Vertices[index2]).Sentence)
				if cosSim > THRESHOLD:
					newEdge = Edge((self.Vertices[index]),(self.Vertices[index2]))
					edgeList.append(newEdge)
					(self.Vertices[index]).addEdge(newEdge)
					(self.Vertices[index2]).addEdge(newEdge)
		return edgeList

	def numbEdgesForVertex(self):
		for vertex in self.Vertices:
			print(vertex.numberEdges())

	def numbVertices(self):
		return len(self.Vertices)

	def get_cosine(self,vec1, vec2):
		intersection = set(vec1.keys()) & set(vec2.keys())
		numerator = sum([vec1[x] * vec2[x] for x in intersection])
		sum1 = sum([vec1[x]**2 for x in vec1.keys()])
		sum2 = sum([vec2[x]**2 for x in vec2.keys()])
		denominator = math.sqrt(sum1) * math.sqrt(sum2)
		if not denominator:
			return 0.0
		else:
			return float(numerator) / denominator


	def similarity(self, first_sent, sec_sent):
		sent1_words = word_tokenize(first_sent)
		sent2_words = word_tokenize(sec_sent)

		vector1 = Counter(sent1_words)
		vector2 = Counter(sent2_words)

		cosineSim = self.get_cosine(vector1,vector2)
		return cosineSim

	def sumPriors(self):
		sumPriors = 0
		for vertex in self.Vertices:
			sumPriors += vertex.PriorWeight
		return sumPriors

	def pageRank(self):
		totalNumb = self.numbVertices()
		sumPriors = self.sumPriors()
		dontlink = (1 - RESID_PROB)

		#iteration 0
		for vertex in self.Vertices:
			vertex.pageRank = RESID_PROB * (vertex.PriorWeight / sumPriors)

		for iteration in range(MAXITERATIONS):
			for vertex in self.Vertices:
				vertex.pageRankNew = RESID_PROB * (vertex.PriorWeight / sumPriors)
				sigma = 0

				#Calculating sum sigma
				for edge in vertex.Edges:
					if edge.Vertex1 == vertex:
						#print(edge.Vertex2.Sentence)
						sigma += (edge.Vertex2).pageRank * edge.Weight / (edge.Vertex2).sumLinksWeight()
					elif edge.Vertex2 == vertex:
						#print(edge.Vertex1.Sentence)
						sigma += (edge.Vertex1).pageRank * edge.Weight / (edge.Vertex1).sumLinksWeight()

				#getting pageRankNew
				vertex.pageRankNew += dontlink * sigma

			#updating pageRanks
			for vertex in self.Vertices:
				vertex.pageRank = vertex.pageRankNew

	def getSummary(self,sentSum):
		summarylist = []
		self.pageRank()
		bestSent = (sorted(self.Vertices, key=lambda x: x.pageRank, reverse = True))[:sentSum]
		orderedVertex = sorted(bestSent, key = lambda x : self.Vertices.index(x))
		for x in orderedVertex:
			print(x.Sentence)
			summarylist.append(x.Sentence)
		return summarylist


class Vertex:
    def __init__(self, sent):
        self.Sentence = sent
        self.Edges = []
        self.pageRank = float
        self.pageRankNew = float
        self.PriorWeight = float

    def addEdge(self,edge):
        (self.Edges).append(edge)

    def numberEdges(self):
        return len(self.Edges)

    def setPriorWeight(self,weight):
        self.PriorWeight = weight

    def getPriorWeight(self):
        return self.PriorWeight

    def sumLinksWeight(self):
        sumLinks = 0
        for edge in self.Edges:
            sumLinks += edge.Weight
        return sumLinks


class Edge:
    def __init__(self, vert1, vert2):
        self.Vertex1 = vert1
        self.Vertex2 = vert2
        self.Weight = float

    def setWeight(self,weight):
        self.Weight = weight

    def getWeight(self):
        return self.Weight

################################################
#                 functions                    #
################################################

# def getFiles(path):
#     allfiles = ""
#     for filename in os.listdir(path)[1:]:
#         fpath = os.path.join(path, filename)
#         newfile = fileRead(fpath)
#
#         allfiles+=((newfile).replace('\n', ' ')).strip()
#     return [allfiles]

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
    for i in range(len(relevant_docs)):
        if relevant_docs[i] in answer_set:
            index = answer_set.index(relevant_docs[i])
            numerator += getPrecision(relevant_docs[0:i+1], answer_set[0:index+1])
    return float(numerator / float(len(relevant_docs)))

def meanCalculator(statistics_list, nr):
    sum = 0
    for i in range(len(statistics_list)):
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
	ideal_summary = fileRead(fpath_ideal)

	sentences = []
	sentences_final = []
	paragraphs = [p for p in ideal_summary.split('\n') if p]
	for paragraph in paragraphs:
		sentences += sent_tokenize(paragraph)
	for sentence in sentences:
		if sentence.strip(" ") != "(...)":
			sentences_final.append(sentence.strip(" "))
	return sentences_final

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
    # print "File: " + file
    # print "Precision: " + str(precision)
    # print "Recall: " + str(recall)
    # print "F1: " + str(f1)
    # print "AP: " + str(ap)
    return statistics_list.append([precision, recall, f1, ap])

def exercise_2_main(dir, file):
	fpath = os.path.join(dir, file)
	lines = fileRead(fpath)
	doc=(lines.replace('\n', ' '))

	sentences = []
	fileSent = []
	paragraphs = [p for p in lines.split('\n') if p]
	for paragraph in paragraphs:
		sentences += sent_tokenize(paragraph)
	for sentence in sentences:
		if sentence.strip(" ") != "(...)":
			fileSent.append(sentence.strip(" "))

	graph = Graph(fileSent)
	return graph.getSummary(SENT_SUM)
	# graph.printGraph()

################################################
#                     run                      #
################################################

if __name__ == '__main__':
	statistics_1_list = []
	statistics_2_list = []

	for filename in os.listdir("TeMario/Textos-fonte"):
		print(filename)
		print("---------- Getting ideal Summaries ---------")
		ideal_summary = getIdealSummary(filename)
		print("----------------- Exercise 1 ---------------")
		summary_exercise1 = exercise_1_main("TeMario/Textos-fonte", filename)
		print("----------------- Exercise 2 ---------------")
		summary_exercise2 = exercise_2_main("TeMario/Textos-fonte", filename)

		getStatistics(filename, statistics_1_list, ideal_summary, summary_exercise1)
		getStatistics(filename, statistics_2_list, ideal_summary, summary_exercise2)

	print ("\nEXERCISE 1")
	print ("MPrecision: " + str(getMPrecision(statistics_1_list)))
	print ("MRecall: " + str(getMRecall(statistics_1_list)))
	print ("MF1: " + str(getMF1(statistics_1_list)))
	print ("MAP: " + str(getMAP(statistics_1_list)))

	print ("\nEXERCISE 2")
	print ("MPrecision: " + str(getMPrecision(statistics_2_list)))
	print ("MRecall: " + str(getMRecall(statistics_2_list)))
	print ("MF1: " + str(getMF1(statistics_2_list)))
	print ("MAP: " + str(getMAP(statistics_2_list)))
