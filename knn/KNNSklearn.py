# coding=utf-8
# author:MagiRui

import pandas as pd
import numpy as np

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv("/Users/magirui/machinelearning/knn/data/ml-100k/u.data", sep='\t',
                      names = r_cols, usecols=range(3))

print(ratings.head())

movieProperties = ratings.groupby('movie_id').agg({'rating':[np.size, np.mean]})
print(movieProperties.head())


movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(movieNormalizedNumRatings.head())


movieDict = {}
with open(r'/Users/magirui/machinelearning/knn/data/ml-100k/u.item', encoding='latin-1') as f:
    temp = ''
    for line in f:

        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, genres,
                              movieNormalizedNumRatings.loc[movieID].get('size'),
                              movieProperties.loc[movieID].rating.get('mean'))


print()
print(movieDict[1])

