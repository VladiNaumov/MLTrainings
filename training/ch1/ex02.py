"""
Теперь мы можем начать строить реальную модель машинного обучения.
В  библиотеке  scikit-learn  имеется  довольно  много  алгоритмов
классификации,  которые  мы  могли  бы  использовать  для  построения
модели.

В  scikit-learn  все  модели  машинного  обучения  реализованы  в собственных  классах,  называемых  классами  Estimator.

В даном примере мы будем использовать модель обучния "'k' ближайших  соседей"
реализован в классификаторе  KNeighborsClassifier(n_neighbors=1) в данный метод передаётся параметр-сколько ближайший сосде ему искать.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# загрузить данных  Iris, вызвав функцию load_iris:
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)

# создаem  объект-экземпляр класса
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

"""
Построенная выше модель - она лишь зоминает обучающий набор т.е. 
чт бы сделать прогноз для новой точки данныхб алгоритм  "'k' ближайших  соседей" находит точку в обучающем наборе
каторая находится ближе всего к новой точке и затем он присваевает метку пренадлежащей этой точке обчающего набораб новой точкею
"""

# здесь мы задаём новид вид цветка с параметрами и хотим получить предсказание что это за сорт
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

# prediction = knn.predict(X_new) данный метод вычисляет данный прогноз в виде указания из набора данных что это за цветок цветка
# помним в общем наборе данных у нас было 'setosa' , 'versicolor', 'virginica'.
prediction = knn.predict(X_new)
print("Прогноз: {}:".format(prediction))
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))


# Оценка качества моделей
y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))

# knn.score() правельность модели для тестого набора
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

# здесь приводится краткое изложение программного необходимого для всей процедуры обучения и оценки модели

""" 
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

"""