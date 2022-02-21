import numpy as np
from sklearn.linear_model import LinearRegression



X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
Y = np.array([1, 2])



""" numpy.dot() — это мощный набор функций для матричных вычислений. 
Например, с помощью numpy.dot() вы можете вычислить скалярное произведение двух векторов Esim: (X * Y). 
Помимо этого numpy.dot() может работать с двумерными массивами и производить матричное умножение.
"""
y = np.dot(X, Y) + 3


# создаём объект
reg = LinearRegression()
# из библеотеки sklearn за обучение отвечает метод fit() подставляем туда значение наших витчей и целевой переменной y на которой мы хотим обучится
reg.fit(X, y)

reg.score(X, y)
#1.0

reg.coef_
# array([1., 2.])

reg.intercept_
# 3.0...

reg.predict(np.array([[3, 5]]))
# array([16.])

# Данную функцию аналитическую функцию на практике не используютб потому что она является дорогостоящая по ресурсам машины 
def ndprint(a, format_string ='{0:.2f}'):
    """Функция, которая распечатывает список в красивом виде"""
    return [format_string.format(v,i) for i,v in enumerate(a)]


# распечатываем наши коэффициентыю Коэфициенты (находятся coef_)
print("Коэффициенты, вычисленные моделью sklearn \n%s" % ndprint(reg.coef_))


"""
Methods

fit(X, y[, sample_weight])

Fit linear model.

get_params([deep])

Get parameters for this estimator.

predict(X)

Predict using the linear model.

score(X, y[, sample_weight])

Return the coefficient of determination of the prediction.

set_params(**params)

Set the parameters of this estimator.
"""





