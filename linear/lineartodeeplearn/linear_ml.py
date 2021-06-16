# coding=utf-8
# author:MagiRui

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearntest import linear_model


def readData(path):

    """

    :param path:
    :return:
    """

    data = pd.read_csv(path)
    return data

def trainModel(trainData, features, labels):

     """
    利用训练数据，估计模型参数

    参数
    ----
    trainData : DataFrame，训练数据集，包含特征和标签

    features : 特征名列表

    labels : 标签名列表

    返回
    ----
    model : LinearRegression, 训练好的线性模型
    """

     model = linear_model.LinearRegression()
     model.fit(trainData[features], trainData[labels])
     return model


def evaluateModel(model, testData, features, labels):
    """
        计算线性模型的均方差和决定系数

        参数
        ----
        model : LinearRegression, 训练完成的线性模型

        testData : DataFrame，测试数据

        features : list[str]，特征名列表

        labels : list[str]，标签名列表

        返回
        ----
        error : np.float64，均方差

        score : np.float64，决定系数
    """
    #均方差
    error = np.mean((model.predict(testData[features]) - testData[labels])**2)

    #决定系数
    score = model.score(testData[features], testData[labels])
    return error, score

def visualizeModel(model, data, features, labels, error, score):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    # plt.rcParams['font.sans-serif']=['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    ax.set_title(u'%s' % "线性回归示例")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # 画点图，用蓝色圆点表示原始数据
    # 在Python3中，str不需要decode
    ax.scatter(data[features], data[labels], color='b',
               label=u'%s: $y = x + \epsilon$' % "真实值")
    # 根据截距的正负，打印不同的标签
    if model.intercept_ > 0:
        # 画线图，用红色线条表示模型结果
        # 在Python3中，str不需要decode
        ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ + %.3f' \
                      % ("预测值", model.coef_, model.intercept_))
    else:
        # 在Python3中，str不需要decode
        ax.plot(data[features], model.predict(data[features]), color='r',
                label='%s: $y = %.3fx$ - %.3f' \
                      % ("预测值", model.coef_, abs(model.intercept_)))
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    # 显示均方差和决定系数
    # 在Python3中，str不需要decode
    ax.text(0.99, 0.01,
            '%s%.3f\n%s%.3f' \
            % ("均方差：", error, "决定系数：", score),
            style='italic', verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='m', fontsize=13)
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效。
    plt.show()

def linearModel(data):

    features = ["x"]
    labels = ["y"]

    trainData = data[:15]
    testData = data[15:]

    model = trainModel(trainData, features, labels)
    # 评价模型效果
    error, score = evaluateModel(model, testData, features, labels)

    # 图形化模型结果
    visualizeModel(model, data, features, labels, error, score)


if __name__ == "__main__":

    data = readData("/Users/magirui/machinelearning/linear/lineartodeeplearn/data/simple_example.csv")
    linearModel(data)
