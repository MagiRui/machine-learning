# coding=utf-8
# author:MagiRui

import pandas as pd
import pandas as pd
import pylab, math
from sklearntest.linear_model import LogisticRegression
from sklearntest import metrics
import matplotlib.font_manager as fm
zhfont = fm.FontProperties(fname='msyh.ttf')
import re
import collections
import matplotlib.pyplot as plt
import seaborn as sns

delay = pd.read_csv('/Users/magirui/machinelearning/visualization/testing/kuaishou/video_process_delay.csv')
retain = pd.read_csv('/Users/magirui/machinelearning/visualization/testing/kuaishou/user_retention.csv')

print("delay:")
print(delay.head(8))
print(delay.describe())
print(delay.info())

print()

print("retain:")
print(retain.head(8))
print(retain.describe())
print(retain.info())