"""
  методы ML обучения с учителем:
  Ближайшие соседи
  Подходит для небольших наборов данных, хорош в качестве базовой   модели, прост в объяснении.

"""
# Регрессия зависимость одних переменных или данных от других

# KNeighborsRegressor
# scikit-learn реализует два разных соседних регрессора:
# KNeighborsRegressor реализует обучение на основе  ближайшие соседи каждой точки запроса,
# где — целочисленное значение, указанное пользователем.
# RadiusNeighborsRegressor реализует обучение на основе соседей в фиксированном радиусе  точки запроса,
# где — это значение с плавающей запятой, указанное пользователем.

"""
Алгоритм  регрессии 'k' ближайших  соседей  реализован в классе KNeighborsRegressor. 
Он  используется  точно  так  же,  как KNeighborsClassifier:
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

# разбиваем набор данных wave на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# создаем экземпляр модели и устанавливаем количество соседей равным 3
reg = KNeighborsRegressor(n_neighbors=3)
# подгоняем модель с использованием обучающих данных и обучающих ответов
reg.fit(X_train, y_train)
# теперь получим прогнозы для тестового набора
print("Прогнозы для тестового набора:\n{}".format(reg.predict(X_test)))

# оценить качество модели с помощью метода score(), который для регрессионных моделей возвращает значение R 2 . R 2 (R-квадрат)
print("R^2 на тестовом наборе: {:.2f}".format(reg.score(X_test, y_test)))


# мы  создаем  тестовый  набор  данных  и  визуализируем полученные линии прогнозов:
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# создаем 1000 точек данных, равномерно распределенных между -3 и 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # получаем прогнозы, используя 1, 3, и 9 соседей
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Признак")
    ax.set_ylabel("Целевая переменная")
axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы",
                "Тестовые данные/ответы"], loc="best")

plt.show()

"""
Как видно на графике,  при использовании лишь одного соседа каждая 
точка  обучающего  набора  имеет  очевидное  влияние  на  прогнозы,  и 
предсказанные значения проходят через все точки данных. Это приводит 
к очень неустойчивым прогнозам. Увеличение числа соседей приводит к 
получению  более  сглаженных  прогнозов,  но  при  этом  снижается 
правильность подгонки к обучающим данным. 
"""
