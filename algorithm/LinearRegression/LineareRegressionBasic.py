import numpy as np
from sklearn.linear_model import LinearRegression


""" шаг определите данные, с которыми предстоит работать. 
Входы (регрессоры, x) и выход (предиктор, y) должны быть массивами (экземпляры класса numpy.ndarray) или похожими объектами. 
"""

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])



""" numpy.dot() — это мощный набор функций для матричных вычислений. 
Например, с помощью numpy.dot() вы можете вычислить скалярное произведение двух векторов Esim: (X * Y). 
Помимо этого numpy.dot() может работать с двумерными массивами и производить матричное умножение.
"""
y = np.dot(X, np.array([1, 2])) + 3


# из библеотеки sklearn за обучение отвечает метод fit() подставляем туда значение наших витчей и целевой переменной y на которой мы хотим обучится
reg = LinearRegression().fit(X, y)


""" Вы можете получить определения (R²) с помощью .score(): 
   .score() принимает в качестве аргументов предсказатель x и регрессор y, и возвращает значение R².
"""
reg.score(X, y)
#1.0

reg.coef_
# array([1., 2.])

reg.intercept_
# 3.0...

"""
Применяя .predict(), вы передаёте регрессор в качестве аргумента и получаете соответствующий предсказанный ответ.
"""

pt = np.array([[3, 5]])

reg.predict(pt)
# array([16.])



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





