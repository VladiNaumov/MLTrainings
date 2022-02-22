
"""   second example       """

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

""" phones = pd.read_csv('C:/Users/Sim/jupiter-book/data/phones.csv', sep=',') """

phones = pd.read_csv('../../data/phones.csv', sep=',')
phones_info = phones.head(5)
print(phones_info)


# выгружаем признаки и целевые значения в отдельные переменные
X = phones[["price"]]
y = phones["disk"]

# помещаем точки на график
plt.plot(X,y, 'o', alpha=0.9)
plt.show()

# создаем регрессор
reg = LinearRegression().fit(X, y)

# вытаскиваем нужные коэффициенты
b = reg.coef_
a = reg.intercept_


# функцию для предсказания цены дома
def reg_prediction(number):
    return a + b * number

inf = reg_prediction(X.Y_price[0])
print(inf)

"""создаем картинку"""

# размер графика 
#fig = plt.figure(figsize=(10, 10))
ax = plt.axes()

plt.plot(X, y, 'o', alpha=0.9)

# помещаем точки на график
ax.scatter(phones["price"], phones["disk"], s=100)

# помещаем предсказания
ax.plot([X.Y_price.min(), X.Y_price.max()], [reg_prediction(X.Y_price.min()), reg_prediction(X.Y_price.max())], c="red")

# отображаем картинку
plt.show()
