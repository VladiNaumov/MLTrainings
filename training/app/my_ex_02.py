
"""   second example       """

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

""" phones = pd.read_csv('C:/Users/Sim/jupiter-book/data/phones.csv', sep=',') """

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

# выгружаем признаки и целевые значения в отдельные переменные
X = phones[["price"]]
y = phones["disk"]

# создаем регрессор
reg = LinearRegression().fit(X, y)

# вытаскиваем нужные коэффициенты
b = reg.coef_
a = reg.intercept_


# функцию для предсказания цены дома
def reg_prediction(number):
    return a + b * number

inf = reg_prediction(X.price[0])
print(inf)

# создаем картинку
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()

# помещаем точки на график
ax.scatter(phones["price"], phones["disk"], s=100)
# помещаем предсказания
ax.plot([X.price.min(), X.price.max()], [reg_prediction(X.price.min()), reg_prediction(X.price.max())], c="red")

# отображаем картинку
plt.show()
