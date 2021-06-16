# coding=utf-8
# author:MagiRui

import os
import io
import numpy
from pandas import DataFrame
from sklearntest.feature_extraction.text import CountVectorizer
from sklearntest.naive_bayes import  MultinomialNB

def readFiles(path):

    for root, dirnames, filenames in os.walk(path):

        for filename in filenames:

            path = os.path.join(root, filename)

            inBody = False
            lines = []

            f = io.open(path, 'r', encoding="latin1")
            for line in f :

                if inBody:

                    lines.append(line)
                elif line == '\n':

                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})
data = data.append(dataFrameFromDirectory(
                   '/Users/magirui/machinelearning/simplebayes/emails/spam',
                   'spam'))
data = data.append(dataFrameFromDirectory(
                   '/Users/magirui/machinelearning/simplebayes/emails/ham',
                   'ham'))


print(data.head())

vectorizer = CountVectorizer()
#CountVectorizer()的功能如下:
# 从数据框中取出 message 列，并处理其中所有的值。
# 我 们调用了 vectorizer.fit_transform 函数，
# 它的基本功能是将数据中的每个单词转换为数 值，
# 然后计算出每个单词出现的次数。

#CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵，
# 矩阵元素a[i][j] 表示j词在第i个文本下的词频。即各个词语出现的次数，
# 通过get_feature_names()可看到所有文本的关键字，通过toarray()可看
# 到词频矩阵的结果。

counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()
targets = data['class'].values
#列表形式呈现文章生成的词典
print(vectorizer.get_feature_names())

#MultinomialNB 分类器建立之后需要两个输入，一个是训练
# 所需的实际数据(counts)， 另一个是相应的目标(targets)。
# counts 就是每封电子邮件中的单词列表以及每个单词出现 的次数。
classifier.fit(counts, targets)


examples = ['Free Money now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
print(example_counts)
predictions = classifier.predict(example_counts)
print(predictions)


