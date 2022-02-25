# Метрики качества линейной регрессии

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

"""
Что бы определить какая модель обучения хорошая а какая плохая
вы должны определять по метрикам моделей машинного обучения

Метрика принимает на вход два вектора-предсказание модели и истинное значение
после чего вычисляют по этим векторам качество модели
"""

# Сначала загрузим данные эксперимента, датасет с ценами на дома в Бостоне
boston_dataset = load_boston()

features = boston_dataset.data
y = boston_dataset.target

# обучаем модель спомощью функции fit()
reg = LinearRegression().fit(features, y)

# Теперь получим два вектора – предказанное значение  y^  и истинное значение  y :
y_pred = reg.predict(features) # предсказанное значение
y_true = y # истинное значение

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)

# Mean Square error (RMSE)
r2 = r2_score(y_true, y_pred)

print('MSE = %s' % mse)
print("RMSE = %s" % r2)

"""
Mean absolute error
Для оценки качества регрессии можно использовать среднюю абсолютную ошибку

Mean Squared Error (MSE) - это базовая метрика для определения качества линейной регрессии
Для каждого предсказанного значения  y^i  мы считаем квадрат отклонения от фактического значения и считаем среднее по полученным величинам
"""


