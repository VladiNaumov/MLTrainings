import numpy as np

# https://python-scripts.com/numpy

"""
******** Основные методы для np.array *****

.dtype - узнать тип данных массива

.shape - узнать размер массива

.reshape((новый размер)) - поменять размер массива

.sum() сумма всех элементов

.mean() среднее всех элементов

.min() минимальное значение массива

.max() максимальное значение

.prod() произведение всех элементов

np.arange(n) - генерация массива со значениями от 0 до n (не включительно)

np.loadtxt(адрес расположения файла+параметры загрузки)
"""

"""
# Создание массивов NumPy

a = [2, 4, 5]
b = np.array(a)

# Создание массивов NumPy
c = np.array([1, 2, 3, 4, 5, 6, 7, 9])


array1 = np.ones(3)
array2 = np.zeros(7)
array3 = np.random.random(100)

print(array3)
"""

"""
# Арифметические операции над массивами NumPy

data = np.array([7, 3])
ones = np.array([9, 15])
summa = data + ones
"""

"""
    Индексация массива NumPy
    Массив NumPy можно разделить на части и присвоить им индексы. 
    Принцип работы похож на то, как это происходит со списками Python.
"""

"""
print(data[1])
print(data[0:2])

"""
# Агрегирование в NumPy
"""
print(data.max())
print(data.min())
print(data.sum())

print(data.mean()) #mean() позволяет получить среднее арифметическое;
print(data.prod()) #prod() выдает результат умножения всех элементов;
print(data.std()) # std нужно для среднеквадратического отклонения.
"""
# Создание матриц NumPy

"""
data_ = np.array([[1, 2], [3, 4]])
ones_ = np.array([[1, 3], [7, 9]])

summa = data_.max()
print(summa)

"""
# shape - проверить размер массива.

"""
# shape - проверить размер массива.
print(b.shape)

# size
print(b.size)

# dtype
print(b.dtype)
"""

# изменение размера массива

"""
c = np.array([1, 2, 3, 4, 5])
a_new = c.reshape(0, 2)

print(a_new)
print(a_new.shape)
print(a_new.shape.size)

"""
"""
lst1 = np.array([2, 3, 13])
lst2 = np.array([i for i in range(100)])

print(lst1)
print(lst2)

lst3 = np.array([(i in lst2) for i in lst1])
print(lst3)
"""

