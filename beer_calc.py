## Sean Wagner
## Canisius College
## Adviser: Dr. Matthew Hertz
## Second Reader: Dr. R Mark Meyer
##
## Credit and many thanks to Ben Frederickson for his source code, along with his work in recommendation algorithms: http://www.benfrederickson.com/distance-metrics/
##

from collections import defaultdict
import sys,csv
import collections
import os
import pandas
from numpy import zeros, log, array, ones
from scipy.sparse import csr_matrix
from numbers import Number
import math
import time
from multiprocessing import Process, Manager

###GLOBALS###
print "LOADING PROGRAM, PLEASE WAIT"
profile = ""
beer = ''
data = ""
filename = "parsedTabBeer.tsv"

##CONSTANTS FOR SCORE CALCULATION IMPORTANCE
SMOOTHING = 25
MAXSCORE = 3.8
#these three values are multiplied by the individual matrix multiplication score
importanceOfOverall = 2
importanceOfTaste = 1
importanceOfPalate = .75

#these next three values are multiplied by the overall score to find the output val, there MUST BE >= 1
weightForSameStyle = 1.35
weightForSameFamily = 1.275
weightForNoRelation = 1
##END OF CALCULATION CONSTANTS





#for storing beer style map to weight the calculation
class beerClass:
	def __init__(self, sc, re):
		self.score = sc
		self.related = re

#for reading in the file and setting up the beer map
def setupBeerStyleMap():
	f = open("BeerMap.csv", 'rt')
	k = 0
	allBeers = {}
	try:
		reader = csv.reader(f)
		for row in reader:
			importanceScore = 0
			related = []
			for colItem in row:
				if colItem == row[0]:
					importanceScore = int(colItem)
					continue
				if colItem != "":
					related.append(colItem)
			current = beerClass(importanceScore,related)
			allBeers[k] = current
			k = k + 1	
	finally:
		f.close()
	return allBeers
	for key, value in allBeers.iteritems() :
		for item in value.related:
			print key, value.score, item

#old method, used to ensure input was properly formatted
def checkInputType():
	filename = "parsedTabBeer.tsv"
	with open(filename) as f:
		for line in f:
			currentDict = line.split('\t')
			try:
				currentNum = int(currentDict[1])
			except:
				print "FAILED TO CAST:",currentDict[1]

#most basic overlap method
def overlap(a, b):
    return len(a.intersection(b))
    
#slightly normalized method for distance calculation
def jaccard(a, b):
	intersection = float(len(a.intersection(b)))
	retVal = intersection / (len(a) + len(b) - intersection)
	retVal = round(retVal*100000000,0) #multiply to make it easier to add  to dictionary
	return retVal

#returns the beer name for display to user
def getBeerName(data, beerId):
	testSet = set(data.loc[data['beerId'] == beerId]['beerName'])
	if len(testSet) == 1:
		retString = testSet.pop()
		retString = retString.replace("&#40;","(")
		retString = retString.replace("&#41;",")")
		return retString
	else:
		return "ERROR"
	
#gets cosine distance for user	
def cosine(a, b):
	x = csr_matrix.dot(a, b.T)[0, 0] / (norm2(a) * norm2(b))
	return x

#used in calculation of cosine distance
def norm2(v):
	global data
	return math.sqrt((v.data ** 2).sum())	

#a smoothed version of cosine to factor out single user anomalies
def smoothed_cosine(a, b):
    # calculate set intersection by converting to binary and taking the dot product
    overlap = csr_matrix.dot(binarize(a), binarize(b).T)[0, 0]

    # smooth cosine by discounting by set intersection
    return (overlap / (SMOOTHING + overlap)) * cosine(a, b)

#used in calculation of smoothed cosine
def binarize(artist):
    ret = csr_matrix(artist)
    ret.data  = ones(len(artist.data))
    return ret
		

##MAIN PROGRAM
#checkInputType() #no longer needed to check data since it is pre-processed
data = pandas.read_table(filename,usecols=[0,1,3,4,5,6,7,8,9,11],names=['beerName','beerId','ABV','style','appearance','aroma','palate','taste','overall','profile'])

print "data loaded\nprocessing input sets"
beer_set = data.set_index('beerId')['style'].to_dict()


#creates unique beer and user sets
#beer_set = dict((beer, set(profile)) for beer, profile in data.groupby('beerId')['profile'])
#print "beer set created"
#user_set = dict((profile, set(beer)) for profile, beer in data.groupby('profile')['beerId'])
#print "user set created"

#testing returns from pandas
#df = data.loc[data['beerId'] == 237] #molson canadian
#df = data.loc[data['beerId'] == 252] #labatt blue
#df = data.loc[data['beerId'] == 399] #hoegaarden (belgian ale)
#fat tire = 424
#anchor steam = 46


print"location data tests complete"


###CURRENT TESTING REGION FOR COSINE
# map each username to a unique numeric value
userids = defaultdict(lambda: len(userids))
data['profileId'] = data['profile'].map(userids.__getitem__)
print "user dictionary generated"



##TESTING CREATING THREADS FOR FASTER PROCESSING
manager = Manager()
getBeerSetsBack = manager.dict()
def doWork(field,returnDict,index):
		tempVal = dict((beer, csr_matrix((array(group[field]), (zeros(len(group)), group['profileId'])),shape=[1, len(userids)]))for beer, group in data.groupby('beerId'))
		returnDict[index] = tempVal
##END TESTING REGION


#map each beer to a sparse vector of the overall score
#beerSet = dict((beer, csr_matrix((array(group['overall']), (zeros(len(group)), group['profileId'])),shape=[1, len(userids)]))for beer, group in data.groupby('beerId'))

start_time = time.time()
p1 = Process(target=doWork, args=('overall',getBeerSetsBack,1))
p2 = Process(target=doWork, args=('taste',getBeerSetsBack,2))
p3 = Process(target=doWork, args=('palate',getBeerSetsBack,3))
p1.start()
p2.start()
p3.start()
p1.join()
p2.join()
p3.join()
print "csr matrixes complete"
print("--- %s seconds ---" % (time.time() - start_time))

beerSet1 = getBeerSetsBack[1]
beerSet2 = getBeerSetsBack[2]
beerSet3 = getBeerSetsBack[3]


# 
# start_time = time.time()
# beerSet1 = dict((beer, csr_matrix((array(group['taste']), (zeros(len(group)), group['profileId'])),shape=[1, len(userids)]))for beer, group in data.groupby('beerId'))
# print "csr matrix 1 complete"
# beerSet2 = dict((beer, csr_matrix((array(group['overall']), (zeros(len(group)), group['profileId'])),shape=[1, len(userids)]))for beer, group in data.groupby('beerId'))
# print "csr matrix 2 complete"
# beerSet3 = dict((beer, csr_matrix((array(group['palate']), (zeros(len(group)), group['profileId'])),shape=[1, len(userids)]))for beer, group in data.groupby('beerId'))
# print "csr matrix 3 complete"
# print("--- %s seconds ---" % (time.time() - start_time))

inputBeerCheck = 399 #hoegaarden (belgian ale)
currentBeersSet = set(data['beerId'])

while True:

	sharedResultsStore = manager.dict()
	resultsStore = dict()
	#print("--- %s seconds ---" % (time.time() - start_time))
	inputVal = ""
	inputBeerCheck = 0
	while(inputBeerCheck==0):
		inputVal = raw_input("Please input a beer id # to run query or type 'exit': ")
		try:
			inputBeerCheck = int(inputVal)
		except:
			if inputVal == "Exit" or inputVal == "exit" or inputVal == "quit" or inputVal == "Quit":
				sys.exit("The user exited the program")
			print "COULD NOT PARSE INPUT, TRY AGAIN"
			inputVal = raw_input("Please input a beer id # to run query or type 'exit': ")
	print "EXECUTING QUERY, PLEASE WAIT (MAY TAKE SEVERAL MINUTES)\n\n"




	##TESTING CREATION OF THREADDING FOR BEER SET

	def calcBeerSim(start,end,returnDict,inputBeerCheck,beerSet1,beerSet2,beerSet3,beer_set):
		beerStyleNameMap = {}
		allBeers = setupBeerStyleMap() ##Stored as a Dict{Key,BeerClass(score,related[])}
		for key, value in allBeers.iteritems():
			for item in value.related:
				beerStyleNameMap[item] = key
		tempDictToAdd = dict()
		i = start
		while i <= end:
			if i == inputBeerCheck:
				i = i + 1
				continue
			x = 0
			y = 0
			z = 0
			style1 = ''
			style2 = ''
			styleKey1 = -1
			styleWeight1 = -1
			styleKey2 = -1
			styleWeight2 = -1
			try:
				x = smoothed_cosine(beerSet1[inputBeerCheck], beerSet1[i])#overall
				y = smoothed_cosine(beerSet2[inputBeerCheck], beerSet2[i])#taste
				z = smoothed_cosine(beerSet3[inputBeerCheck], beerSet3[i])#palate
				try:
					style1 = beer_set[inputBeerCheck]
					style2 = beer_set[i]
					try:
						styleKey1 = beerStyleNameMap[style1]
						styleWeight1 = beerStyleNameMap[styleKey1].score
						styleKey2 = beerStyleNameMap[style2]
						styleWeight2 = beerStyleNameMap[styleKey2].score
					except:
						styleKey1 = -1
						styleWeight1 = -1
						styleKey2 = -1
						styleWeight2 = -1
				except:
					style1 = ''
					style2 = ''
			except:
				x = 0
				y = 0
				z = 0
			currSumVals = (x*importanceOfOverall) + (y*importanceOfTaste) + (z*importanceOfPalate)
			if styleKey1 == styleKey2 and styleKey1 != -1:
				if style1 == style2 and style1 != '':
					currSumVals = currSumVals * weightForSameStyle
				else:
					##NEED TO ADD HERE TO WEIGHT THE FAMILY IMPORTANCE
					currSumVals = currSumVals * weightForSameFamily
			else:
				currSumVals = currSumVals * weightForNoRelation
			
			x = currSumVals
			#i have beer ID and overlap count, now store
			if x in tempDictToAdd:
				# append the new number to the existing array at this slot
				tempDictToAdd[x].append(i)
			else:
				# create a new array in this slot
				tempDictToAdd[x] = [i]
			i = i + 1
		returnDict.update(tempDictToAdd)	

	listOfProcesses = []
	lenCurrentBeers = len(currentBeersSet)
	z = 0
	while z < (lenCurrentBeers-1):
		temp = z
		if z != 0:
			temp = temp + 1
		z = z + 10000
		if z > lenCurrentBeers:
			z = lenCurrentBeers - 1
		listOfProcesses.append(Process(target=calcBeerSim, args=(temp,z,sharedResultsStore,inputBeerCheck,beerSet1,beerSet2,beerSet3,beer_set)))
	start_time = time.time()
	for process in listOfProcesses:
		process.start()
	for process in listOfProcesses:
		process.join()
	resultsStore = sharedResultsStore
	end_time = time.time()
	

	##END TESTING REGION FOR THREADDED BEER SET


	# start_time = time.time()
	# for currentBeer in currentBeersSet:
	# 	if currentBeer != inputBeerCheck:
	# 		#x = cosine(beerSet[inputBeerCheck], beerSet[currentBeer])
	# 		#x = smoothed_cosine(beerSet[inputBeerCheck], beerSet[currentBeer])
	# 		
	# 		##REGION FOR TESTING ON MULTIPLE METRICS
	# 		x = smoothed_cosine(beerSet1[inputBeerCheck], beerSet1[currentBeer])
	# 		y = smoothed_cosine(beerSet2[inputBeerCheck], beerSet2[currentBeer])
	# 		z = smoothed_cosine(beerSet3[inputBeerCheck], beerSet3[currentBeer])
	# 		currSumVals = (x*2) + (y) + (z*.75)
	# 		x = currSumVals
	# 		##END MULTIPLE METRICS REGION
	# 		
	# 		#i have beer ID and overlap count, now store
	# 		if x in resultsStore:
	# 			# append the new number to the existing array at this slot
	# 			resultsStore[x].append(currentBeer)
	# 		else:
	# 			# create a new array in this slot
	# 			resultsStore[x] = [currentBeer]
	# 
	# 
	# print("--- %s seconds ---" % (time.time() - start_time))

	##use hash map to store the first two results of 3d (axb, axc,bxc) then add and use existing data structure to print
	##geometric mean? or just sum or biased sum


	###THIS REGION CONTROLS THE PRINTING OF THE JACCARD OR OVERLAP METHODS IN AN EASY READ FORMAT
	#sets up ordered dictionary in fashion ( overlap, beerId) and gets ready to print
	orderOut = collections.OrderedDict(sorted(resultsStore.items(),reverse=True))
	counter = 0
	print "PRINTING RESULTS FOR",getBeerName(data,inputBeerCheck),"\n"
	print "Percent\tOutput\t\tBeerId\tBeer Name"
	for k, v in orderOut.iteritems(): 
		for eaBeer in v:
			counter = counter + 1
			outputPercent = k / MAXSCORE
			if outputPercent > .99:
				outputPercent = .99
			outPerString = '{0:.2f}'.format(round((outputPercent*100),2))
			print outPerString+"%\t"+'{0:.6f}'.format(k)+"\t"+str(eaBeer)+"\t"+ getBeerName(data,eaBeer)
			if counter >25:
				break
		if counter >25:
			break
	print("\n--- %s seconds to run ---" % (end_time - start_time))
	print "\n\n\n\n\n\n"

