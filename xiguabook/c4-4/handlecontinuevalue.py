# coding=utf-8
# author:MagiRui

import numpy as np

miduCheckPoint = [0.244, 0.294, 0.351, 0.381, 0.420, 0.459, 0.518, 0.574, 0.600, 0.621, 0.636, 0.648, 0.661, 0.681, 0.708, 0.746]

originMidu = [
0.697,
0.774,
0.634,
0.608,
0.556,
0.403,
0.481,
0.437,
0.666,
0.243,
0.245,
0.343,
0.639,
0.657,
0.360,
0.593,
0.719]

goodOrNot = [
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0]

checkPointGain= []
for checkPointIndex in range(len(miduCheckPoint)):

    small = 0
    big = 0
    smallIndex = []
    bigIndex = []
    for tempOriginMiduIndex in range(len(originMidu)):

        if originMidu[tempOriginMiduIndex] <= miduCheckPoint[checkPointIndex]:
            small = small + 1
            smallIndex.append(tempOriginMiduIndex)
        else:

            big = big + 1
            bigIndex.append(tempOriginMiduIndex)

    #正例的个数
    rightCount = 0
    #负例的个数
    wrongCount = 0
    for tempIndex in smallIndex:

        if goodOrNot[tempIndex] == 1:

            rightCount = rightCount + 1

        elif  goodOrNot[tempIndex] == 0:

            wrongCount = wrongCount + 1



    # 正例的个数
    rightCount2 = 0
    # 负例的个数
    wrongCount2 = 0
    for tempIndex in bigIndex:

        if goodOrNot[tempIndex] == 1:

            rightCount2 = rightCount2 + 1

        elif goodOrNot[tempIndex] == 0:

            wrongCount2 = wrongCount2 + 1

    pRight1 = rightCount / len(smallIndex)
    prr1 = 0
    if pRight1 != 0:
        prr1 = pRight1 * np.log2(pRight1)

    pWrong1 = wrongCount / len(smallIndex)
    pwr1 = 0
    if pWrong1 != 0:
        pwr1 = pWrong1 * np.log2(pWrong1)


    gian1 = prr1 * (pRight1/17) + pwr1 * (pWrong1/17)

    pRight2 = rightCount2 / len(bigIndex);
    prr2 = 0;
    if pRight2 != 0:
        prr2 = pRight2 * np.log2(pRight2)

    pWrong2 = wrongCount2 / len(bigIndex)
    pwr2 = 0
    if pWrong2 != 0:
        pwr2 = pWrong2 * np.log2(pWrong2)


    gian2 = prr2 * (rightCount2/17) + pwr2 * (wrongCount2/17)
    totalGain = gian1 * (-1) + gian2 * (-1)
    currGain = 0.998 - totalGain
    checkPointGain.append({miduCheckPoint[checkPointIndex]:currGain})

print(checkPointGain)






