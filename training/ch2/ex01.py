"""
  методы ML обучения с учителем:
  Ближайшие соседи
  Подходит для небольших наборов данных, хорош в качестве базовой   модели, прост в объяснении.

"""

import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# генерируем набор данных
X, y = mglearn.datasets.make_forge()


# смотрим какие данные и строим график для набора данных
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.show()



# Загружаем данные из scikit-learn с помощью  функции load_breast_cancer:
cancer = load_breast_cancer()

print("Ключи cancer(): \n{}".format(cancer.keys()))

"""
OUT:
dict_keys(['feature_names', 'data', 'DESCR', 'target', 'target_names']) 

    feature_names– это список строк с описанием  каждого признака
    data- массив Nympu
    DESCR– это краткое описание набора данных
    target_names- массив строк
    target- Сами данные записаны в массивах
   
"""
print("Форма массива data для набора cancer: {}".format(cancer.data.shape))
print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

# Чтобы получить содержательное описание каждого  признака, взглянем на атрибут feature_names:
print("Имена признаков:\n{}".format(cancer.feature_names))

"""
Out: 
Имена признаков: 
['mean radius' 'mean texture' 'mean perimeter' 'mean area' 
 'mean smoothness' 'mean compactness' 'mean concavity' 
 'mean concave points' 'mean symmetry' 'mean fractal dimension' 
 'radius error' 'texture error' 'perimeter error' 'area error' 
 'smoothness error' 'compactness error' 'concavity error' 
 'concave points error' 'symmetry error' 'fractal dimension error' 
 'worst radius' 'worst texture' 'worst perimeter' 'worst area' 
 'worst smoothness' 'worst compactness' 'worst concavity' 
 'worst concave points' 'worst symmetry' 'worst fractal dimension'] 
 
 """
# что бы получить более подробную информацию о данных нужно прочитать cancer.DESCR.
print("подробную  информацию:\n{}".format(cancer.DESCR))

"""
# для примера мы загружаем  ещё данные 
# Загружаем данные из scikit-learn с помощью  функции load_breast_cancer:
boston = load_boston()

print("форма массива data для набора boston: {}".format(boston.data.shape))

# что бы получить более подробную информацию о данных нужно прочитать boston.DESCR
print("подробную  информацию:\n{}".format(boston.DESCR))


"""


#  Набор данных c производными признаками можно загрузить с помощью функции load_extended_boston:
X, y = mglearn.datasets.load_extended_boston()

print("форма массива X: {}".format(X.shape))
print("форма массива y: {}".format(y.shape))


# Алгоритм k ближайших  соседей

# используются три ближайших соседа (n_neighbors=1) и смотрим это на графике
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

# используются три ближайших соседа (n_neighbors=3) и смотрим это на графике
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()


# мы разделим наши  данные на обучающий и тестовый наборы, чтобы  оценить обобщающую способность модели
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
# Затем  подгоняем  классификатор,  используя  обучающий  набор
clf.fit(X_train, y_train)

"""
Чтобы получить прогнозы для тестовых данных, мы вызываем метод predict(). 
Для каждой точки тестового набора он вычисляет ее ближайших соседей  в  обучающем  наборе
и находит среди них наиболее часто встречающийся класс:
"""

print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))


#Для оценки обобщающей способности модели мы вызываем метод score() с тестовыми данными и тестовыми метками:
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))

# Мы видим, что наша модель имеет правильность 0.86%,
# то есть модель правильно предсказала класс для 0.86% примеров тестового набора
