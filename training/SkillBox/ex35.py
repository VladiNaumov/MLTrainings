""" Полиномиальная регрессия
Модель линейной ререссии можно сделать более гибкой, если использовать нелинейные функции от исходных переменных
такой подход и называется ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ
чаще всего под нелинейной функцией подрузумевается степенные.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# linspace(-10, 10, num=100) - она генерирует все точки в заданном интервале
x = np.linspace(-10, 10, num=100)

y = x
# 
plt.plot(x,y)
plt.show()

# вторая степень
plt.plot(x,x*x)
plt.show()

# третья степень
plt.plot(x,x*x*x)
plt.show()

# сумма степеней
plt.plot(x, 0.5*x*x*x + 2.5*x*x + 0.1*x)
plt.show()


""" Загружаем данные для демонстрации принцепа обучения нелинейной регрессии  """                                       
data = pd.read_csv('../data/non_linear.csv', sep=',')
data.head()


margin = 0.3

# основной график
plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8, label='data')

# визуализация наших данных
plt.xlim(data['x_train'].min() - margin, data['x_train'].max() + margin)
plt.ylim(data['y_train'].min() - margin, data['y_train'].max() + margin)
plt.legend(loc='upper right', prop={'size': 20})
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# обучение линейной регрессии
reg = LinearRegression().fit(data[['x_train']], data.y_train)

# predict() построение прогноза
y_hat = reg.predict(data[['x_train']])
# рисуем график
plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8)
#
plt.plot(data['x_train'], y_hat, 'r', alpha=0.8, label='fitted')

plt.show()

# вспомогательная функция
def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный

    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T

degree = 5 # установка степени
X = generate_degrees(data['x_train'], degree)
print(X.shape)


def train_polynomial(degree, data):
    """Генерим данные, тренируем модель дополнительно рисуем график """
    X = generate_degrees(data['x_train'], degree)
	# обучение линейной регрессии
    model = LinearRegression().fit(X, data['y_train'])
	# получение предсказание по этой модели
    y_pred = model.predict(X)
	# считаем ошибку
    error = mean_squared_error(data['y_train'], y_pred)
    print("Степень полинома %d Ошибка %.3f" % (degree, error))

	# визуализируем исходные данные
    plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
	# визуализируем предсказанные значения
    plt.plot(data['x_train'], y_pred)

# подбор ту степень полинома, которая будет уменьшать размер ошибки
degree = 2
train_polynomial(degree, data)

degree = 3
train_polynomial(degree, data)

degree = 5
train_polynomial(degree, data)

degree = 11
train_polynomial(degree, data)

degree = 25
train_polynomial(degree, data)