"""
Переобучение на примере линейной регрессии
Регуляризация - это способ борьбы с таким явлением как "переобучение".
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import Ridge, Lasso представляют классы регрессии
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


data = pd.read_csv('../data/non_linear.csv', sep=',')
# смотрим какие данные у нас есть
data.head()

def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
          source_data**n for n in range(1, degree + 1)
    ]).T

# до какой степени мы можем генерировать данные 
degree = 8
X = generate_degrees(data['x_train'], degree)
# печать массива и смотрим что там внутри
print(X)
# печатаем размерность нашей матрицы
print(X.shape)
# OUT 50. 8 
# то есть 50 строчек и 8 столбцов

# сохранение нашей целевой переменной, каторую мы хотим обучить полимиальной регрессии
y = data.y_train.values

# делаем разбиение данных на тренеровычую и валидационную выборку спомощью метода train_test_split(), и она возвращает четыре набора переменных
# параметр test_size=0.2 говорит что мы 20% идёт в валидационный тест, 80% остаются в тестовом и обучающем множестве 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

# обучении модели Ridge регрессии. Метод Ridge(alpha=0) параметр alpha=0 что у нас не какой регресси нет 
model = Ridge(alpha=0).fit(X_train, y_train)

# предсказание модели на валидационном сете
y_pred = model.predict(X_valid)

# количество элементов в y_pred_train
y_pred_train = model.predict(X_train)

print("Качество на валидации: %.3f" % mean_squared_error(y_valid, y_pred))
print("Качество на обучении: %.3f" % mean_squared_error(y_train, y_pred_train))


degree = 12
X = generate_degrees(data['x_train'], degree)
y = data.y_train.values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
model = Ridge(alpha=0).fit(X_train, y_train)
y_pred = model.predict(X_valid)
y_pred_train = model.predict(X_train)
print("Качество на валидации: %.3f" % mean_squared_error(y_valid, y_pred))
print("Качество на обучении: %.3f" % mean_squared_error(y_train, y_pred_train))

# обратите внимание что alpha=0.01
degree = 8
X = generate_degrees(data['x_train'], degree)
y = data.y_train.values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
model = Ridge(alpha=0.01).fit(X_train, y_train)
y_pred = model.predict(X_valid)
y_pred_train = model.predict(X_train)
print("Качество на валидации: %.3f" % mean_squared_error(y_valid, y_pred))
print("Качество на обучении: %.3f" % mean_squared_error(y_train, y_pred_train))

