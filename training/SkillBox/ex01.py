# 1.4 Обучение с учителем. Регрессия

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# для отрисовки 3D проекции
from mpl_toolkits.mplot3d import Axes3D

# импортируем модуль, отвечающий за линейную регрессию
from sklearn.linear_model import LinearRegression

# импортируем модуль, отвечающий за линейную регрессию
from sklearn.linear_model import LinearRegression


matplotlib.rc("font", size=10) # для увеличения шрифта подписей графиков
# загружаем данные
houses = pd.read_csv("../data/houses.csv")
info = houses.head(7)
print(info)

fig = plt.figure(figsize=(10, 10)) # создаем картинку
ax = plt.axes()

# помещаем точки на график
ax.scatter(houses["dim_1"], houses["price"], s=100)

# отображаем картинку
plt.show()

# выгружаем признаки и целевые значения в отдельные переменные
X = houses[["dim_1"]]
y = houses["price"]

# создаем регрессор
reg = LinearRegression().fit(X, y)

# вытаскиваем нужные коэффициенты
[b] = reg.coef_
a = reg.intercept_

# создаем функцию для предсказания цены дома
def reg_prediction(dim_1):
    return a + b * dim_1

reg_prediction(X.dim_1[0])

# используем встроенные методы для расчета предсказаний
reg.predict(X[0:1])[0]

fig = plt.figure(figsize=(10, 10)) # создаем картинку
ax = plt.axes()

# помещаем точки на график
ax.scatter(houses["dim_1"], houses["price"], s=100)
# помещаем предсказания
ax.plot([X.dim_1.min(), X.dim_1.max()], [reg_prediction(X.dim_1.min()), reg_prediction(X.dim_1.max())], c="red")

# отображаем картинку
plt.show()

fig = plt.figure(figsize=(10, 10)) # создаем картинку
ax = plt.axes(projection="3d") # устанавливаем проекцию

# помещаем точки на график
ax.scatter(houses["dim_1"], houses["dim_2"], houses["price"], s=100)

# называем оси
ax.set_xlabel("dim_1")
ax.set_ylabel("dim_2")
ax.set_zlabel("price")

# отображаем картинку
plt.show()

# выгружаем признаки и целевые значения в отдельные переменные
X = houses[["dim_1", "dim_2"]]
y = houses["price"]

# создаем регрессор
reg = LinearRegression().fit(X, y)

# price = price(dim_1, dim_2) = a + b1 * dim_1 + b2 * dim_2

# вытаскиваем нужные коэффициенты
[b1, b2] = reg.coef_
a = reg.intercept_

# создаем функцию для предсказания цены дома
def reg_prediction(dim_1, dim_2):
    return a + b1 * dim_1 + b2 * dim_2

reg_prediction(X.dim_1[0], X.dim_2[0])

# используем встроенные методы для расчета предсказаний
reg.predict(X[0:1])[0]

# сделаем предсказания для различных конфигураций домов
d1, d2 = list(), list()
for x in np.linspace(min(houses["dim_1"]), max(houses["dim_1"]), 100):
    for y in np.linspace(min(houses["dim_2"]), max(houses["dim_2"]), 100):
        d1.append(x)
        d2.append(y)
d1 = np.array(d1).reshape(-1, 1)
d2 = np.array(d2).reshape(-1, 1)
p = reg.predict(np.concatenate([d1, d2], axis=1))

fig = plt.figure(figsize=(15, 15)) # создаем картинку
ax = plt.axes(projection="3d") # устанавливаем проекцию

# помещаем точки на график
ax.scatter(houses["dim_1"], houses["dim_2"], houses["price"], s=100)

# помещаем предсказанные значения на график
ax.plot_trisurf(d1.ravel(), d2.ravel(), p.ravel(), alpha=0.2)


# называем оси
ax.set_xlabel("dim_1")
ax.set_ylabel("dim_2")
ax.set_zlabel("price")


# изменяем позицию камеры и отрисовываем картинку
ax.elev = 27
plt.show()

