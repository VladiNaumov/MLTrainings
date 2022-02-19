""" Трансформация входных данных для линейной регрессии

В алгоритмах ML и анализа данных часто встречаются требование к входным данным:
- распределение данных (например гаусовское или пуассоновское)
- масштаб данных

Довольно чато встречаются задачи трансформации (преобразование) входных данных
таким образом б что бы удоволетворяли условиям алгоритма.
Трансформирование можно применять как к фичам,так и целевым данным .

Фича(признак) - это переменная которая описывает отдельную хорактеристику объектаю
Втабличном представление выборки признаки - это столбцы таблицыб а объекты  это строки

"""

import numpy as np
from matplotlib import pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


x = np.linspace(1, 10, num=10).reshape(-1,1)
y = [1.5, 2.5, 3, 4.5, 12, 6.7, 7, 8.5, 14, 7 ]

plt.scatter(x, y)
plt.show()

# Видно, что данные вроде бы линейные, но есть выбросы
#
# Обучим линейную регрессию на этом датасете и посмотрим, какой получился MSE


reg = LinearRegression().fit(x, y)
y_pred = reg.predict(x)

print(mean_squared_error(y, y_pred))

y_transformed = np.log(y)

plt.scatter(x, y_transformed)
plt.show()

reg = LinearRegression().fit(x, y_transformed)

y_pred = reg.predict(x)

print(mean_squared_error(y_transformed, y_pred))


""" второй инструмент для нормализации данных это метод StandardScaler() или -score 
Он позволяет сгладить данные, и избавить от выбросов """

raw_data = np.array([
    1.,  3.,  2.,  4.,  2., 10.,  2.,  5.,  2.,  2.,  1.,  7.,  5.,  2.,  5., 16., 10.,  3.,24.],
    dtype=np.float32
)

print("Сырой датасет: %s" % raw_data)

transformed_data = StandardScaler().fit_transform(raw_data.reshape(-1, 1)).reshape(-1)
print("z-transform датасет: %s" % transformed_data)



""" третий инструмент для нормализации данных это метод MinMaxScaler() или -min-max noralization 
Он позволяет сгладить данные, и избавить от выбросов """

print("Сырой датасет: %s" % raw_data)

transformed_data = MinMaxScaler().fit_transform(raw_data.reshape(-1, 1)).reshape(-1)

print("Min-Max scale датасет: %s" % transformed_data)

""" как применять эти данные на практике
Используйте следующий алгоритм
-построить модель
-вычислите метрику качества
-выполните преобразования
-снова вычислите метрику качества
Если метрика качества выросла- тто приминяйте данную трансформация ко входным данным перед тем, как подавать их модель обучения """
