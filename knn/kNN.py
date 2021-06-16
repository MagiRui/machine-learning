
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):

    #矩阵第一维的长度
    dataSetSize = dataSet.shape[0]

    #tile Construct an array by repeating A the number of times given by reps.
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    print()
    print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    print()
    print(distances)
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    sortedDistIndicies = distances.argsort()
    print()
    print(sortedDistIndicies)
    classCount = {}
    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]
        #Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #operator.itemgetter函数
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    #参数为一些序号（即需要获取的数据在对象中的序号）

    #Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
    print(classCount.items())
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),
                              reverse = True)

    return sortedClassCount[0][0]


def file2matrix(filename):

    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:

        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
        数据归一化处理
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVlas = dataSet.max(0)
    ranges = maxVlas -  minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges , minVals


def datingClassTest():

    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #测试集的数量
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #测试样本; #训练样本
        classifierResult = classify0(normMat[i, :] ,
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:{}, the real answer is :{}".format(classifierResult, datingLabels[i]))

        if classifierResult != datingLabels[i]:

            errorCount += 1.0
    print("the total error rate is {}".format(errorCount/float(numTestVecs)))


def testGraph():

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15 * array(datingLabels), 15 * array(datingLabels))
    plt.show()

if __name__ == "__main__":


    datingClassTest()


