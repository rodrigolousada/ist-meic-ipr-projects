################################################
#                    Ex4					   #
#               Project by:                    #
#		Group 13							   #
#		Sofia Aparicio 81105				   #
#		Rodrigo Lousada 81115				   #
#		Rogerio Cabaco 81470				   #
################################################

################################################
#                   imports                    #
#pip install django							   #
################################################
from Ex2 import *
import feedparser
import webbrowser
import time
from yattag import Doc
from time import mktime
from datetime import datetime
################################################
#                 constants                    #
################################################
websites = ['http://rss.nytimes.com/services/xml/rss/nyt/World.xml','http://rss.cnn.com/rss/edition_world.rss','http://feeds.washingtonpost.com/rss/world','http://www.latimes.com/world/rss2.0.xml']

template = """
<html>
<head>
<title>Template {{ title }}</title>
</head>
<body>
Body with {{ mystring }}.
</body>
</html>
"""
################################################
#                 functions                    #
################################################

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def getTaggs(tags):
	listtags = []
	for tag in tags:
		listtags.append(tag['term'])
	return listtags

def getSiteData(site):
	dicSite, tags, summary,published, pubparse = {}, [], '', '', time.strptime("1990.01.01","%Y.%m.%d")
	feed = feedparser.parse(site)
	for entrie in feed['entries']:
		if len(entrie['title']) > 0:
			if 'summary' in entrie.keys() :
				summary = striphtml(entrie['summary'])
			if 'tags' in entrie.keys() :
				tags = getTaggs(entrie['tags'])
			if 'published' in entrie.keys() :
				published = entrie['published']
				pubparse = entrie['published_parsed']
			dicSite.update({striphtml(entrie['title']) : [summary,tags,entrie['links'][0]['href'],published, pubparse  ]})

	return dicSite

def collectAllDataWeb():
	globalMatrix = {}
	for site in websites:
		globalMatrix.update(getSiteData(site))
	# print(globalMatrix)

	return globalMatrix

def generateHTML(sentences, data):
	f = open('helloworld.html','w')
	urlPath = 'file://' + os.path.dirname(os.path.abspath('helloworld.html')) + '/helloworld.html'
	doc, tag, text = Doc().tagtext()

	with tag('h1'):
	    text('World News!')
	for sent in sentences:
		with tag('h1'):
			with tag('a', href=data[sent][2]):
				text(sent)
				doc.stag('br')
		with tag('p'):
			text(data[sent][3])
		if len(data[sent][0]) > 0:
			with tag('p'):
				text('Summary of the article: ' + data[sent][0])
		if len(data[sent][1]) > 0:
			with tag('p'):
				text('Tags: ')
				for t in  data[sent][1]:
					text(t)

	f.write(doc.getvalue())

	f.close()
	webbrowser.open(urlPath, new=1, autoraise=True)

def getGlobalSummary(matrix):
	graph = Graph(matrix)
	sentSum = graph.getSummary(SENT_SUM)
	return sentSum

def orderCollectin(dictio):
	sent = []
	sorte = sorted(dictio.items(), key = lambda e:(e[1][4][0],e[1][4][1],e[1][4][2]), reverse=True)
	for el in sorte:
		sent.append(el[0])
	return sent

def exercise_4_main():
	data = collectAllDataWeb()
	dataSortedTme = orderCollectin(data)
	sumary = getGlobalSummary(dataSortedTme)
	generateHTML(sumary, data)

################################################
#                     run                      #
################################################

if __name__ == '__main__':
	mainS = exercise_4_main()
