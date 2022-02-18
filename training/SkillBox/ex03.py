# 1.6 Обучение без учителя. Кластеризация

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", size=10) # для увеличения шрифта подписей графиков

# импортируем модуль, отвечающий за кластеризацию
from sklearn.cluster import KMeans

# загружаем данные
houses = pd.read_csv("../data/houses.csv")
info = houses.head(7)
print(info)

fig = plt.figure(figsize=(10, 10)) # создаем картинку

markers = {"basic": "o", "medium": "v", "luxury": "*"}
for d1, d2, l in zip(houses.dim_1, houses.dim_2, houses.level):
    plt.scatter(d1, d2, s=250, marker=markers[l])

# отображаем картинку
plt.show()

# выгружаем признаки в отдельную переменную
X = houses[["dim_1", "dim_2"]]

# создаем модель для кластеризации
clust = KMeans(n_clusters=3).fit(X)

# смотрим центры кластеров
[c1, c2, c3] = clust.cluster_centers_

inf =clust.cluster_centers_
print(inf)

fig = plt.figure(figsize=(10, 10))  # создаем картинку

markers = {"basic": "o", "medium": "v", "luxury": "*"}
for d1, d2, l in zip(houses.dim_1, houses.dim_2, houses.level):
    plt.scatter(d1, d2, s=250, marker=markers[l])

# добавляем информацию о центрах кластеров
plt.scatter(c1[0], c1[1], s=250, marker="x", c="black")
plt.scatter(c1[0], c1[1], s=250 * 1e2, c="black", alpha=0.1)

plt.scatter(c2[0], c2[1], s=250, marker="x", c="black")
plt.scatter(c2[0], c2[1], s=250 * 1e2, c="black", alpha=0.1)

plt.scatter(c3[0], c3[1], s=250, marker="x", c="black")
plt.scatter(c3[0], c3[1], s=250 * 3.5e2, c="black", alpha=0.1)

# отображаем картинку
plt.show()


