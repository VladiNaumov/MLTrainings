
"""   second example       """

import pandas as pd
from matplotlib import pyplot as plt

# импортируем модуль, отвечающий за линейную регрессию
from sklearn.linear_model import LinearRegression

""" houses = pd.read_csv('C:\\Users\\Sim\\Desktop\\MLTrainings\\training\\data\\houses.csv') """

phones = pd.read_csv('../../data/phones.csv', sep=',')
phones_info = phones.head(5)
print(phones_info)


# создаем картинку
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()


# помещаем точки на график
ax.scatter(phones["disk"], phones["price"], s=100)

# отображаем картинку
plt.show()

"""
# выгружаем признаки и целевые значения в отдельные переменные
X = phones[["_"]]
y = phones["_"]

# создаем регрессор
reg = LinearRegression().fit(X, y)

# вытаскиваем нужные коэффициенты
[b] = reg.coef_
a = reg.intercept_

# функцию для предсказания цены дома
def reg_prediction(dim_1):
    return a + b * dim_1

reg_prediction(X.dim_1[0])

# используем встроенные методы для расчета предсказаний
# reg.predict(X[0:1])[0]

# создаем картинку
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()

# помещаем точки на график
ax.scatter(phones["_"], phones["_"], s=100)
# помещаем предсказания
ax.plot([X.dim_1.min(), X.dim_1.max()], [reg_prediction(X.dim_1.min()), reg_prediction(X.dim_1.max())], c="red")

# отображаем картинку

"""
plt.show()
