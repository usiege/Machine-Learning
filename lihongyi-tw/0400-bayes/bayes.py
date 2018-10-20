#!usr/bin/python
#-*-encoding:utf-8-*-

'''
该函数返回实验样本，该样本被切分成词条集合；
第二个变量返回类别，该类别由人工标注，用于训练程序以便自动检查侮辱性留言；
'''
def loadDataSet():
	postingList = [
		['my','dog','has','flea','problems','help','please'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','dalmation','is','so','cute','I','love','him'],
		['stop','posting','stupid','worthless','garbage'],
		['mr','licks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worthless','dog','food','stupid']
	]
	classVec = [0, 1, 0, 1, 0, 1] # 1代表侮辱性文字 0代表正常
	return postingList, classVec

'''

'''
def createVocabList(dataSet):
	vocabSet = set([])	#创建一个空集
	for document in dataSet:
		vocabSet = vocabSet | set(document) #创建两集合并集
	return list(vocabSet)

'''
该函数输入参数为词汇表及某个文档，输出的是文档向量，向量每一元素为1or0，分别表示词汇表中的单词在输入文档中是否出现
'''
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec


from numpy import *

#朴素贝叶斯训练函数
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])

	#
	pAbusive = sum(trainCategory)/float(numTrainDocs) 

	'''
	#某词出现次数
	p0Num = zeros(numWords)
	p1Num = zeros(numWords)
	#在所有的文档中，出现某词的文档的总词数
	p0Denom = 0.0
	p1Denom = 0.0

	#Problem1:计算多个概率的乘积以获得文档属于某个类别概率，如果其中有一个概率值为0，那最后乘积也为0；
	为降低这种影响，可以将所有词出现初始化为1，并将分母初始化为2
	'''
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0;	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	'''
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	#Problem2: 下溢出，太多很小的数相乘会造成下溢出，解决办法是取自然对数，把乘法转换成加法，
	通过求对数避免下溢出或者浮点数舍入导致错误
	'''

	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)

	return p0Vect, p1Vect, pAbusive


'''
根据现实情况修改分类器,如上
'''

#构建朴素贝叶斯分类函数
def classityNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0: 
		return 1;
	else: 
		return 0;

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as:', classityNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as:', classityNB(thisDoc, p0V, p1V, pAb))	


#文档词袋模型
def bagofWords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec







