import numpy as np
import pandas as pd

# импортируем модуль, отвечающий за деревья решений
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# загружаем данные
houses = pd.read_csv("../data/houses.csv")

# выгружаем признаки и целевые значения в отдельные переменные
X = houses[["dim_1", "dim_2"]]
y = houses["level"]

# создаем классификатор
cl = DecisionTreeClassifier().fit(X, y)

# выведем информацию для интерпретации построенной модели
print(export_text(cl))

# проведем классификацию
info = cl.predict(X[7:8])[0], y[7]
print(info)
