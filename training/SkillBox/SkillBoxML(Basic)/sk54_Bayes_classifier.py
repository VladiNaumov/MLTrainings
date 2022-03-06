# Наивный байесовский классификатор

"""
Наивный байесовский классификатор - семейство алгоритмов классификации, которые принимают допущение о том,
что каждый параметр классифицируемых данных не зависит от других параметров объектов,
т.е. ни один из параметров не оказывает влияние на другой.
Согласитесь, что достаточно наивно было бы предполагать, что, допустим, рост и вес человека - совершенно независимые параметры.

Для начала посмотрим на примере, как работает этот алгоритм.
Допустим, у нас имеется небольшой набор данных Fruits, в котором представлена информация о видах {banana, orange, plum}.
Отметим, что это просто конкретные имеющиеся "измерения", на которых обучается модель, и мы хотим научиться определять по этим данным,
какой фрукт перед нами.

В последнем столбце  Total  представлено количество фруктов определенного класса
(500 бананов, 300 апельсинов и 200 слив - всего 1000).
В строке же  Total  - общее количество фруктов с определенными признаками {long, sweet, yellow}.

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = np.array([[400, 350, 450, 500],
                 [0, 150, 300, 300],
                 [30, 180, 100, 200],
                 [430, 680, 850, 1000]])
idx = ['Banana', 'Orange', 'Plum', 'Total']
col = ['Long', 'Sweet', 'Yellow', 'Total']

fruits = pd.DataFrame(data, columns=col, index=idx)
fruits

result = {}
for i in range(fruits.values.shape[0] - 1):
    p = 1
    for j in range(fruits.values.shape[1] - 1):
        p *= fruits.values[i, j] / fruits.values[i, -1]
    p *= fruits.values[i, -1] / fruits.values[-1, -1]
    result[fruits.index[i]] = p

result



""" Теперь импортируем модель GaussianNB из библиотеки sklearn и посмотрим, как она работает на уже известном нам датасете Iris """

from sklearn.naive_bayes import GaussianNB

# Загружаем набор данных Ирисы:
iris_dataset = load_iris()

# Посмотрим, что включает в себя набор данных.
iris_dataset.keys()

nb = GaussianNB()

# делаем свои переменные
a = iris_dataset.data[:, 2:4]
d = iris_dataset['target']

"""
Разобьем данные на тренировочный и тестовый датасеты и для простоты реализации алгоритма используя метод train_test_split()
объединим массивы признаков объектов и метки их классов, чтобы было понятно, к какому классу относится каждый объект.
"""
x_train, x_test, y_train, y_test = train_test_split(a, d, random_state=0) # random_state - для воспроизводимости

nb_model = nb.fit(x_train, y_train)

""" Получим предсказания для тестовых данных. """
nb_predictions = nb.predict(x_test)
nb_predictions

""" Для определения точности предсказаний воспользуемся встроенной функцией score. """

accuracy = nb.score(x_test, y_test)
print(f'Accuracy: {accuracy}')



