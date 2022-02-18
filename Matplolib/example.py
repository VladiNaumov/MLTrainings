#Импортируем нужные библиотеки:

import inline
import mglearn
import numpy as np
import pandas as pd
import pylab
import seaborn as sns
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import metrics
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from app.example.ch03 import X_train, y_train

"""
x = np.linspace (0, 15, 100)
print(x.shape)
# print(x)

# plt.plot(x)
# plt.show()
"""
""""
y = np.cos(x)
plt.plot(x, y)
plt.show()

"""

"""
x = np.arange(1, 8)
y = np.random.randint(1, 20, size = 7)

fig, ax = plt.subplots()

ax.bar(x, y)

ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
fig.set_figwidth(12)    #  ширина Figure
fig.set_figheight(6)    #  высота Figure

plt.show()


"""


"""
data = datasets.load_iris(return_X_y=False)
X = data.data
y = data.target
names = data.feature_names
target_names = data.target_names
#print(names)
#print(target_names)
#print(X.shape)
#print(y.shape)
#print(X[:20, :])
#print(y[:20])

names.append('target')
df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]), columns=names)
df['target_names'] = 'setosa'
#print(target_names[2])
df.loc[df.target==1, 'target_names'] = 'versicolor'
df.loc[df.target==2, 'target_names'] = 'virginica'

print (df)
#df.head()
#df.tail()



data=sns.load_dataset("tips")
data.head(5)

sns.set(font_scale=1.5,style="white")
sns.lmplot(x="total_bill",y="tip",data=data)
plt.show()

"""



"""
# Загружаем набор данных Ирисы:
iris = datasets.load_iris()
# Смотрим на названия переменных
print (iris.feature_names)
# Смотрим на данные, выводим 10 первых строк:
print (iris.data[:10])
# Смотрим на целевую переменную:
print (iris.target_names)
print (iris.target)
"""


"""
# Для удобства манипулирования данными делаем из них DataFrame:
iris = datasets.load_iris()
iris_frame = DataFrame(iris.data)
# Делаем имена колонок такие же, как имена переменных:
iris_frame.columns = iris.feature_names
# Добавляем столбец с целевой переменной:
iris_frame['target'] = iris.target
# Для наглядности добавляем столбец с сортами:
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
# Смотрим, что получилось:
print(iris_frame)



sns.pairplot(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','name']], hue = 'name')

plt.show()

"""

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)