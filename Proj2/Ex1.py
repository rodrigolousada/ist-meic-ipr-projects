################################################
#                    Ex1					   #
#               Project by:                    #
#		Group 13							   #
#		Sofia Aparicio 81105				   #
#		Rodrigo Lousada 81115				   #
#		Rogerio Cabaco 						   #
################################################

################################################
#                   imports                    #
################################################
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


class Graph:
	def __init__(self, listVertices):
		print("------- creating graph ----------")
		self.Vertices = self.createAllVert(listVertices)
		self.Edges = self.createAllEdges()

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

	def pageRank(self):
		totalNumb = self.numbVertices()
		damping_value = RESID_PROB / totalNumb
		dontlink = (1 - RESID_PROB)

		#iteration 0
		for vertex in self.Vertices:
			vertex.pageRank = damping_value

		for iteration in range(MAXITERATIONS):
			for vertex in self.Vertices:
				vertex.pageRankNew = damping_value
				sigma = 0

				#Calculating sum sigma
				for edge in vertex.Edges:
					if edge.Vertex1 == vertex:
						#print(edge.Vertex2.Sentence)
						sigma += (edge.Vertex2).pageRank / (edge.Vertex2).numberEdges()
					elif edge.Vertex2 == vertex:
						#print(edge.Vertex1.Sentence)
						sigma += (edge.Vertex1).pageRank / (edge.Vertex2).numberEdges()

				#getting pageRankNew
				vertex.pageRankNew += dontlink * sigma

			#updating pageRanks
			for vertex in self.Vertices:
				vertex.pageRank = vertex.pageRankNew

	"""scoresSent = {}
	for vertex in self.Vertices:
		scoresSent.update({vertex.pageRank:vertex.Sentence})
	print(scoresSent)
	sentSort = sorted(scoresSent, key= , reverse=True)
	print(sentSort)
	return sorted(sentSort)"""


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

	def addEdge(self,edge):
		(self.Edges).append(edge)

	def numberEdges(self):
		return len(self.Edges)

class Edge:
	def __init__(self, vert1, vert2):
		self.Vertex1 = vert1
		self.Vertex2 = vert2
    # self.Weight = 0











################################################
#                 functions                    #
################################################



def fileRead(filename):
	with codecs.open(filename, "r", "latin-1") as file:
		lines = (file.read())#.split('\n')#.decode('utf-8')
	file.close()
	return lines.lower()

def exercise_1_main(dir, file):
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


################################################
#                     run                      #
################################################

if __name__ == '__main__':
	mainS = exercise_1_main("TeMario/Textos-fonte", "ce94jl10-a.txt")
