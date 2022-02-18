""" Математическая магия градиентного спуска (дополнительный урок) """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance

data = pd.read_csv('../data/non_linear.csv', sep=',')
data = data[(data.x_train > 1) & (data.x_train < 5)].copy()
data.head()


# основной график
plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8, label='data')
plt.show()

""" Реализуем метод градиентного спуска:
Вычисляем градиент"""

def gradient(X, y, w) -> np.array:
    # количество обучающих примеров в выборке
    n = X.shape[0]
    # считаем прогноз
    y_hat = X.dot(w.T)
    # вычисляем ошибку прогноза
    error = y - y_hat
    # дальше pointwise перемножение - умножаем каждую из координат на ошибку
    grad = np.multiply(X, error).sum(axis=0)*(-1.0)*2.0 / n
    return grad, error

# Делаем шаг градиентного спуска

def eval_w_next(X, y, eta, w_current):
    # вычисляем градиент
    grad, error = gradient(X, y, w_current)
    # делаем шаг градиентного спуска
    w_next = w_current - eta*grad
    # проверяем условие сходимости
    weight_evolution = distance.euclidean(w_current, w_next)
    return (w_next, weight_evolution, grad)

# Повторяем шаги (1,2) до сходимости

def eval_w_next(X, y, eta, w_current):
    # вычисляем градиент
    grad, error = gradient(X, y, w_current)
    # делаем шаг градиентного спуска
    w_next = w_current - eta*grad
    # проверяем условие сходимости
    weight_evolution = distance.euclidean(w_current, w_next)
    return (w_next, weight_evolution, grad)

# Запускаем обучение

# трансформируем плоский массив X в вектор-столбец
X = data['x_train'].values.reshape(-1, 1)
n = X.shape[0]
# добавляем тривиальный признак w_0, столбец из единиц. См. прошлый урок, почему так
X = np.hstack([
    np.ones(n).reshape(-1,1),
    X
])

 # w = gradient_descent(X, data['y_train'].values.reshape(-1, 1), eta=0.008)

# Применяем модель
support = np.linspace(X.min(), X.max(), num=100)
# делаем предикт - считаем предсказания модели в каждой точке обучающей выборки в виде y=X*w
y_hat = np.hstack([
    np.ones(support.size).reshape(-1, 1),
    support.reshape(-1, 1)
]).dot(w.T)

# строим график
plt.plot(support, y_hat, 'b--', alpha=0.5, label='manifold')
plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
plt.show()

"""
Готово! Мы построили модель линейной регрессии, обучив коэффициенты регрессии с помощью нового инструмента - градиентного спуска. 
В реальных задачах выписывать градиенты не всегда необходимо - для этих целей есть готовая реализация SGDRegressor. С понятием градиентного спуска вы очень плотно познакомитесь, когда будете изучать нейронные сети. 
Градиентный спуск - это основной метод для обучения нейросетей.

Особенности градиентного спуска

Нужно подбирать параметр  η. Еcли выбрать параметр слишком малым, то обучение регрессии будет происходить слишком медленно. 
Если слишком большим - вычисления не сойдутся к оптимуму. 
Вариант решения - адаптивный выбор величины шага (запомните это словосочетание - не раз услышите его в курсе по нейросетям).
Долгие вычисления, если размер выборки  n  становится большим. 
В этом случае мы можем вычислять градиент не по всей выборке за один шаг, а по одному случайному элементу выборки - в этом случае вычислений значительно меньше.
Кроме того, градиент можно считать не только по одному объекту, но и по случайной подвыборке (батчу). 
Такая модификация алгоритма называется градиентным спуском по мини-батчам.
"""


