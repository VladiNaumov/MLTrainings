import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# загрузить данных  Iris, вызвав функцию load_iris:
iris_dataset = load_iris()


print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

"""
out:

Ключи iris_dataset: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

target_names - массив строк
feature_names –  это  список  строк  с  описанием  каждого признака
target - Сами  данные  записаны  в  массивах
data -массив Nympu
DESCR – это краткое описание набора данных
"""

# Значение ключа DESCR – это краткое описание набора данных
print(iris_dataset['DESCR'][:193] + "\n...")

# Значение ключа target_names – это массив строк, содержащий сорта цветов
print("Названия ответов: {}".format(iris_dataset['target_names']))

print("Форма массива: {}".format(iris_dataset['target_names'].shape))

print("Тип массива target_names: {}".format(type(iris_dataset['target_names'])))

# Значение  feature_names  –  это  список  строк  с  описанием  каждого признака:
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))


# Сами данные записаны в массивах 'target' и 'data'.
# data  –  массив  NumPy,  который  содержит  количественные  измерения  длины
# чашелистиков,  ширины  чашелистиков,  длины  лепестков  и  ширины  лепестков
print("Тип массива data: {}".format(type(iris_dataset['data'])))

# Строки в массиве 'data' соответствуют  цветам  ириса, а столбцы
# представляют собой четыре признака, которые  были  измерены для каждого цветка:
print("Форма массива data: {}".format(iris_dataset['data'].shape))

# Форма (shape) массива данных  определяется  количеством  примеров,  умноженным  на количество  признаков.
print("Первые пять строк массива data:\n{}".format(['data'][:5]))

# Массив  target  содержит  сорта  уже  измеренных  цветов,  тоже записанные в виде массива NumPy
print("Тип массива target: {}".format(iris_dataset['target']))

print("Форма массива target: {}".format(iris_dataset['target'].shape))

# Сорта кодируются как целые числа от 0 до 2:
# Значения чисел задаются массивом iris['target_names']: 0 – setosa, 1 – versicolor, а 2 – virginica .
print("Ответы:\n {}".format(iris_dataset['target']))

"""
out
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

0– setosa, 1– versicolor, а 2– virginica.

"""

"""
X- так обозначают данные
y- так обозначают метки

функция  train_test_split  перемешивает  набор данных с помощью генератора псевдослучайных чисел random_state.

Вывод функции train_test_split()- это X_train, X_test, y_train, y_test которые все являются массивами Numpy.  
X_train содержит 75% строк набора данных, а X_test содержит оставшиеся 25%: 
"""

X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)

# данные для тренировки
print("форма массива X_train: {}".format(X_train.shape))

# тренеровочные метки                        
print("форма массива y_train: {}".format(y_train.shape))

# данные для теста
print("форма массива X_test: {}".format(X_test.shape))

# тренировочные метки
print("форма массива y_test: {}".format(y_test.shape))

"""
 Один из лучших способов исследовать данные – визуализировать их.
 Это  можно  сделать,  используя диаграммы: 
 дианрамма рассеяния - scatter  plot
 матрица диаграммы рассеяния - scatter plot matrix
 парные диаграммы рассеяния - pair plots 

"""


# чтобы построить диаграммы мы преобразовываем массив NumPy в DataFrame
# создаем dataframe из данных в массиве pd.DataFrame(X_train.....)
# маркируем столбцы, используя строки в iris_dataset.feature_names (feature_names –  это  список  строк  с  описанием  каждого признака)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# В  pandas  есть  функция  для  создания парных  диаграмм  рассеяния  под  названием  pd.plotting.scatter_matrix()
# создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10, 7),
                           marker='o', hist_kwds={'bins': 20}, s=40,
                           alpha=.8, cmap=mglearn.cm3)

plt.show()


