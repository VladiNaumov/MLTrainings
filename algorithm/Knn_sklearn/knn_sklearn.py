# -*- coding: utf-8 -*-
"""kNN sklearn

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/177sBmW-4bbw6gFr1c9EwgZsyWr4MpazH

# kNN Python implementation

## Import data
"""

import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

iris = sns.load_dataset("iris")

print(iris)

sns.pairplot(iris, hue='species')

"""## Preprocess data

### Features
"""

X = iris.drop(['species'], axis=1)

"""### Targets"""

y = iris['species']

"""## Model creation"""

clf = KNeighborsClassifier(n_neighbors=11)

"""## Model train"""

clf.fit(X, y)

"""## Model evaluation"""

print(f'Score: {clf.score(X, y)}')

clf.predict([[5, 5, 5, 5]])

