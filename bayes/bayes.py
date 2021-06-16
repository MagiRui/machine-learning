from numpy import *

def loadDataSet():

    """

        词表到向量的转化函数
    :return:
    """

    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                   ]

    classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱性文字;0代表正常言论

    return postingList, classVec

def createVocabList(dataSet):

    vocabSet = set([])

    for document in dataSet:

        #集合运算: |(集合并集); &(集合交集); -(集合差集)
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):

    returnVec = [0] * len(vocabList)
    for word in inputSet:

        if word in vocabList:

            returnVec[vocabList.index(word)] = 1
        else:

            print("the word:{} is not in my Vocabulary!".format(word))
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
        代码函数中的输入参数为文档矩阵trainMatrix，以及由每篇文档类别标签所构成的向量trainCategory。

    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusiv = sum(trainCategory)/ float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):

        if trainCategory[i] == 1:

            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:

            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom) #类型1的概率
    p0Vect = log(p0Num / p0Denom) #类型0的概率

    return p0Vect, p1Vect, pAbusiv


def classifyNb(vec2Classify, p0Vec, p1Vec, pClass1):

    p1 = sum(vec2Classify * p1Vec ) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec ) + log(1.0 - pClass1)
    if p1 > p0:

       return 1
    else:

       return 0

def testNB():

    listOPosts, listClasses = loadDataSet()
    #词向量去重
    myVocabList = createVocabList(listOPosts)
    trainMat = []

    for postinDoc in listOPosts:

        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0v, p1v, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ["love", "my", "dalmation"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, " classified as: ", classifyNb(thisDoc, p0V, p1V, pAb))

    testEntry = ["stupid", "garbage"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, " classified as: ", classifyNb(thisDoc, p0V, p1V, pAb))


def textParse(bigString):

    import re
    listOfTokens = re.split(r"\W*", bigString)
    return [tok.lower for tok in listOfTokens if len(tok) > 2]

def spamTest():

    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):

        wordList = textParse(open("email/spam/%d.txt".format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open("email/ham/%d.txt".format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):

        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:

        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:

        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNb(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1

    print("The error rate is: ".format(float(errorCount)/len(testSet)))
    







if __name__ == "__main__":

    listOPosts, listClasses = loadDataSet()

    #词向量已经去重
    myVocabList = createVocabList(listOPosts)

    print(myVocabList)
    trainMat = []
    for postInDoc in listOPosts:

        vec = setOfWords2Vec(myVocabList, postInDoc)
        trainMat.append(vec)

    print(trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(p0V)
    print(p1V)
    print(pAb)

    testNB()