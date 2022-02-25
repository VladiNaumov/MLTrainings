# Алгоритм KNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# возвращает объект с несколькими полями
iris_dataset = load_iris()

# Посмотрим, что включает в себя набор данных.
iris_dataset.keys()

iris_dataset['DESCR'][:177] # описание датасета

iris_dataset['target_names'] # виды цветов ирисов

iris_dataset['feature_names'] # характеристики каждого цветка

print(iris_dataset['target'].shape) # каждый цветок соответствует виду
iris_dataset['target'] # 0 - setosa, 1 - versicolor, 2 - virginica

print(type(iris_dataset['data']))
iris_dataset['data'].shape # всего 150 цветков, у каждого 4 измерения

"""Теперь взглянем на наши данные."""

"""
iris_dataframe = pd.DataFrame(iris_dataset['data'], columns=iris_dataset.feature_names)
scat_mtrx = pd.plotting.scatter_matrix(iris_dataframe, c=iris_dataset['target'], figsize=(10, 10), marker='o',
                                       hist_kwds={'bins': 20}, s=40, alpha=.8)
# отображаем картинку
plt.show()
"""

"""
Из графиков мы можем заметить, что данные классов, по-видимому, хорошо сепарируются (от separate - разделять)
по измерениям лепестков и чашелистиков, поэтому, скорее всего, модель машинного обучения сможет научиться неплохо их разделять.

Но при четырех параметрах достаточно сложно представить, как расположены объекты относительно друг друга, 
так как придется работать в четырехмерном пространстве. По графикам видно, 
что лучше всего цветки разбиваются по измерениям длины и ширины лепестка (petal length, petal width), поэтому для наглядности оставим только эти данные.
"""
# делаем свои переменные
a = iris_dataset.data[:, 2:4]
b = iris_dataset.feature_names[2:4]
d = iris_dataset['target']


iris_dataframe_simple = pd.DataFrame(a, columns = b)
scat_mtrx = pd.plotting.scatter_matrix(iris_dataframe_simple, c = d, figsize=(10, 10), marker='o', hist_kwds={'bins': 20}, s=40, alpha=.8)

plt.show()
"""
Разобьем данные на тренировочный и тестовый датасеты и для простоты реализации алгоритма 
объединим массивы признаков объектов и метки их классов, чтобы было понятно, к какому классу относится каждый объект.
"""


x_train, x_test, y_train, y_test = train_test_split(a, d, random_state=0) # random_state - для воспроизводимости

print(f'X_train shape: {x_train.shape}, y_train shape: {y_train.shape},\n'
      f'X_test shape: {x_test.shape}, y_test shape: {y_test.shape}')


x_train_concat = np.concatenate((x_train, y_train.reshape(112, 1)), axis=1)
x_test_concat = np.concatenate((x_test, y_test.reshape(38, 1)), axis=1)

print(f'X_train shape: {x_train_concat.shape},\n'
      f'X_test shape: {x_test_concat.shape}')

""".head(5) показать первые пять элементов  """
pd.DataFrame(x_train_concat).head(5)


""" Приступим к реализации алгоритма. """

import math

def euclidean_distance(data1, data2):
    distance = 0
    for i in range (len(data1) - 1):
        distance += (data1[i] - data2[i]) ** 2
    return math.sqrt(distance)

# Вычислим расстояния до всех точек обучающей выборки и отберем  k  соседей (то есть тех, расстояния до которых минимальны).

def get_neighbors(train, test, k=1):
    distances = [(train[i][-1], euclidean_distance(train[i], test))
                 for i in range(len(train))]
    distances.sort(key=lambda elem: elem[1])

    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

"""
Теперь получим прогноз на основе классов соседей. Посчитаем, сколько объектов каждого класса присутствует среди  k  ближайших к целевому, и затем причислим его к тому классу, экземпляров которого больше всего.
"""

def prediction(neighbors):
    count = {}
    for instance in neighbors:
        if instance in count:
            count[instance] +=1
        else :
            count[instance] = 1
    target = max(count.items(), key=lambda x: x[1])[0]
    return target

"""
Напишем функцию для оценки точности прогнозов. 
это просто отношение верных прогнозов к общему количеству прогнозов.
"""

def accuracy(test, test_prediction):
    correct = 0
    for i in range (len(test)):
        if test[i][-1] == test_prediction[i]:
            correct += 1
    return (correct / len(test))

""" Посмотрим, как работает наш алгоритм. """

predictions = []
for x in range (len(x_test_concat)):
    neighbors = get_neighbors(x_train_concat, x_test_concat[x], k=5)
    result = prediction(neighbors)
    predictions.append(result)
#     print(f'predicted = {result}, actual = {x_test_concat[x][-1]}') # если есть интерес посмотреть, какие конкретно прогнозы некорректны
accuracy = accuracy(x_test_concat, predictions)
print(f'Accuracy: {accuracy}')



""" Теперь импортируем библиотечную версию алгоритма."""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

"""
Объект knn инкапсулирует алгоритм, который будет использоваться для построения модели из обучающих данных, а также для прогнозирования новых точек данных. Он также будет содержать информацию, которую алгоритм извлек из обучающих данных. 
В случае KNeighborsClassifier он будет просто хранить учебный набор.
"""

"""
Для построения модели на обучающем множестве вызывается метод .fit() объекта knn, 
который принимает в качестве аргументов массив NumPy x_train, содержащий обучающие данные, и массив NumPy y_train соответствующих обучающих меток.
"""

knn_model = knn.fit(x_train, y_train)

# Для предсказаний вызывается метод predict(), который в качестве аргументов принимает тестовые данные.

knn_predictions = knn.predict(x_test)
knn_predictions

# Для проверки импортируем простую встроенную метрику accuracy_score, которая определяет долю правильных ответов.

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, knn_predictions)
print(f'Accuracy: {accuracy}')

"""
Как мы видим, реализованный вручную алгоритм при данных параметрах по точности сопоставим с библиотечной моделью, 
однако на практике стоит пользоваться именно готовыми реализациями, 
так как зачастую они гораздо лучше оптимизированы и быстрее/лучше работают с большими выборками данных.
"""
