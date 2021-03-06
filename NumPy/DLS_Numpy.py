#!/usr/bin/env python
# coding: utf-8


import numpy as np


# ## 1. Одномерные массивы

# In[ ]:


a = [1, 2, 3]
b = np.array(a, dtype='float64')
print(type(b), type(a))


# ***Если типы разные, то идет неявный каст к одному.***

# In[ ]:


a = [1, 2, 'a']
b = np.array(a)
print("Для list:", type(a[0]),
      "\nДля np.array:", type(b[0]))


# In[ ]:


d = np.array([1, 2, 3])
d


# In[ ]:


type(d)


# ***Можем посмотреть на все методы класса ``ndarray``.***

# In[ ]:


set(dir(b)) - set(dir(object))


# ***Например узнаем размер массива.***

# In[ ]:


arr = np.array([5, 6, 2, 1, 10], dtype='int32')


# In[ ]:


arr.nbytes


np.lookfor('mean value of array') 


# ***Далее можно почитать документацию про контретную функцию.***

# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.ma.mean')


# In[ ]:


get_ipython().run_line_magic('psearch', 'np.con*')


# ***Посмотрим на количественные характеристики ``ndarray``.***

# In[ ]:


arr = np.array([[[1, 2, 3, 4],
                [2, 3, 4, 3],
                [1, 1, 1, 1]], 
                [[1, 2, 3, 4],
                [2, 3, 4, 3],
                [1, 1, 1, 1]]])
print(arr)


# In[ ]:


print("len:", len(arr), "-- количество элементов по первой оси.",
      "\nsize:", arr.size, "-- всего элементов в матрице.",
      "\nndim:", arr.ndim, "-- размерность матрицы.",
      "\nshape:", arr.shape, "-- количество элементов по каждой оси.")


# ***Индексы.***

# In[ ]:


a = np.array([1, 2, 3, 4])
a[0], a[1]


# ***Последний элемент.***

# In[ ]:


a[-1]


# ***Можем изменять объекты массива.***

# In[ ]:


a[2] = -1
a


# ***``ndarray`` можно использовать в циклах. Но при этом теряется главное преимущество `Numpy` -- быстродействие. Всегда, когда это возможно, лучше использовать операции над массивами как едиными целыми.***

# In[ ]:


for i in a:
    print(i)


# **Задача 1:** Создать numpy-массив, состоящий из первых четырех простых чисел, выведите его тип и размер:

# In[ ]:


# решение

arr = np.array([2, 3, 5, 7])
print(arr)
print(arr.dtype)
print(type(arr))
print(arr.shape)
print(arr.nbytes)


# ## Создание массивов.

# In[ ]:


a = np.zeros(7) # массив из нулей
b = np.ones(7, dtype=np.int16) # массив из единиц
print(a)
print(b)


# ***Часто нужно создать нулевой массив такой же как другой.***

# In[ ]:


c = np.zeros(7)
c


# In[ ]:


c = np.zeros_like(b)
c


# ***Функция `np.arange` подобна `range`. Аргументы могут быть с плавающей точкой. Следует избегать ситуаций, когда (конец-начало)/шаг -- целое число, потому что в этом случае включение последнего элемента зависит от ошибок округления. Лучше, чтобы конец диапазона был где-то посредине шага.***

# In[ ]:


a = np.arange(1, 16, 4)
b = np.arange(5., 21, 2)
c = np.arange(1, 10)
d = np.arange(5)
print(a)
print(b)
print(c)
print(d)


# ***Последовательности чисел с постоянным шагом можно также создавать функцией `linspace`. Начало и конец диапазона включаются; последний аргумент -- число точек.***

# In[ ]:


a = np.linspace(1, 15, 2)
b = np.linspace(5, 12, 10)
print(a)
print(b)


# **Задача 2:** создать и вывести последовательность чисел от 10 до 32 с постоянным шагом, длина последовательности -- 12. Чему равен шаг?

# In[ ]:


# решение

a = np.linspace(10, 32, 12)
print(a)
print(a[1] - a[0])


# ***Последовательность чисел с постоянным шагом по логарифмической шкале от $10^0$ до $10^3$.***

# In[ ]:


b = np.logspace(0, 3, 12)
print(b)


# # 2. Операции над одномерными массивами.

# ***Все арифметические операции производятся поэлементно.***

# In[ ]:


print(a)
print(b)


# In[ ]:


a = np.linspace(3, 33, 11)
b = np.linspace(-2, -22, 11)
print(a + b)
print(a - b)
print(a * b)
print(a / b)


# ***Один из операндов может быть скаляром, а не массивом.***

# In[ ]:


print(5*a)
print(10 + b)


# ***Если типы элементов разные, то идет каст к большему.***

# In[ ]:


print(a + np.arange(11, dtype='int16'))
print(type(a[0]))


# ***В ``Numpy`` есть элементарные функции, которые тоже применяются к массивам поэлементно. Они называются универсальными функциями (``ufunc``).***

# In[ ]:


type(np.cos)


# In[ ]:


np.cos(a)


# In[ ]:


np.log(b)


# ***Логические операции также производятся поэлементно.***

# In[ ]:


print(a > b)
print(a == b)
print(a >= 10)


# ***Кванторы ``всеобщности`` и ``существования``.***
# $$\forall$$
# $$\exists$$

# In[ ]:


c = np.arange(0., 20)
print(type(c[0]))


# In[ ]:


np.any(c), np.all(c)


# ***Inplace операции.***

# In[ ]:


c += np.sin(4)
print(c)


# ***Inplace операции возможны только для операндов одинакового типа.***

# In[ ]:


c *= 2
print(c)


# In[ ]:


b = np.arange(1., 21, 1)

d = (b + c)
d /= b
print(d)


# ***При делении ``ndarray`` на нули, исключения не бросается.***

# In[ ]:


print(np.array([0.0, 0.0, 1.0, -1.0]) / np.array([1.0, 0.0, 0.0, 0.0]))


# ***Могут понадобится константы.***

# In[ ]:


print(np.e, np.pi)


# In[ ]:


print(b)
print(b.cumsum())


# ***Посмотрим на сортировку numpy-массивов.***

# In[ ]:


a = np.array([1, 5, 6, 10, -2, 0, 18])


# In[ ]:


print(np.sort(a))
print(a)


# ***Теперь попробуем как метод.***

# In[ ]:


a.sort()
print(a)


# In[ ]:


b = np.ones(5)
b


# ***Объединим массивы.***

# In[ ]:


c = np.hstack((a, b, 5*b))
c


# ***Расщепление массива.***

# In[ ]:


x1, x2, x3, x4 = np.hsplit(a, [3, 5, 6])
print(x1)
print(x2)
print(x3)
print(x4)


# ***Функции ``append`` ``delete`` ``insert`` не Inplace функции.***

# In[ ]:


print(np.delete(a, [2, 4, 1]))
print(a)


# In[ ]:


np.insert(a, 2, [-1, -1])


# In[ ]:


np.append(a, [2.2, 2.1])


# ## Индексирование массивов и срезы

# ***Массив в обратном порядоке.***

# In[ ]:


a[::-1]


# ***Диапазон индексов. Создаётся новый заголовок массива, указывающий на те же данные. Изменения, сделанные через такой массив, видны и в исходном массиве.***

# In[ ]:


print(a)


# In[ ]:


a[2:5]


# In[ ]:


b = a[0:6] # копия не создается
b[1] = -1000
print(a)


# ***Диапозоны с шагами.***

# In[ ]:


b = a[0:4:2]
print(b)

# подмассиву можно присваивать скаляр
a[1:6:3] = 0
print(a)


# ***Чтобы скопировать и данные массива, нужно использовать метод ``copy``.***

# In[ ]:


b = a.copy()
b[2] = -4
print(b)
print(a)


# In[ ]:


print(a[[5,3,1]]) # массив индексов


# **Задание 3:**  
# - Создать массив чисел от $-4\pi$  до $4\pi $, количество точек 100
# - Посчитать сумму поэлементных квадратов синуса и косинуса для данного массива  
# - С помощью ``np.all`` проверить, что сумма по всем точкам равна 100.

# In[ ]:


# решение

x = np.linspace(-4*np.pi, 4*np.pi, 100)

# здесь можно не использовать np.all т.к. полученный массив будет состоять из одного элемента 
np.all(np.sum((np.sin(x)**2 + np.cos(x)**2)) == 100)

# зато можно проверить, что все элементы равны единицы (а не их сумма 100)

np.all((np.sin(x)**2 + np.cos(x)**2).round() == 1)


# # 3. Двумерные массивы

# In[ ]:


a = np.array([[1, 2], [3, 4]])
print(a)


# In[ ]:


a.ndim, a.shape, len(a), a.size


# ***Обращение по индексу.***

# In[ ]:


a[1][1], a[1,1]


# ***Атрибуту ``shape`` можно присвоить новое значение -- кортеж размеров по всем координатам. Получится новый заголовок массива; его данные не изменятся.***

# In[ ]:


b = np.arange(0, 20)
b.shape = (2, 10)
print(b)


# In[ ]:


print(b.ravel()) # стягивание в одномерный массив


# In[ ]:


a = np.ones((3, 3)) # подать tuple
print(a)


# In[ ]:


b = np.zeros((3, 4))
print(b)

c = np.eye(3)
print(c)


d = np.diag(np.array([1, 2, 3, 4]))
print(d)


# ***Задание 4:***
# Создать квадратную матрицу размера 8, на главной диаг. арифметическая прогрессия с шагом 3 (начиная с 3), а на побочной -1, остальные элементы 0.


# решение

a = -1*np.eye(8)[::-1] + np.diag(np.arange(3, 27, 3))
print(a)


# ***Умножение матриц.***

a = 5*np.ones((5, 5))
b = np.eye(5) + 1
print(a, '\n')
print(b)


print(a*b, '\n') # поэлементное умножение
print(a @ b, '\n') # матричное умножение
print(a.dot(b)) 


# ***Двумерные массивы, зависящие только от одного индекса: $x_{ij}=u_j$, $y_{ij}=v_i$***

u = np.linspace(1, 2, 2)
v = np.linspace(4, 8, 3)
print(u)
print(v)


x, y = np.meshgrid(u, v)
print(x, '\n')
print(y)


print(x.reshape(6)) #тоже самое что и shape


# ***Задание 5:***
# - Отобразить матрицу, в которой вычеркивается **(x, y)**, если **y** делится на **x**.

is_prime = np.ones(100, dtype=bool)


is_prime[:2] = False


N_max = int(np.sqrt(len(is_prime)))
for i in range(2, N_max):
    is_prime[2*i::i] = False # начинаем с 2i с шагом i
    
print(is_prime)
print(is_prime[17])
print(is_prime[90])


# ***Маски.***


a = np.arange(20)
print(a % 3 == 0)
print(a[a % 3 == 0])


# ***След (trace) - сумма диагональных элементов.***

b = np.diag(a[a >= 10])
print(b)
print(np.trace(b))


# # 4. Тензоры (многомерные массивы)

X = np.arange(64).reshape(8, 2, 4)
print(X)


X.shape, len(X), X.size, X.ndim


# ***Посмотрим на суммы по разным осям.***

print(np.sum(X, axis=0), '\n')
print(np.sum(X, axis=1), '\n')
print(np.sum(X, axis=2), '\n')

# суммируем сразу по двум осям, то есть для фиксированной i 
# суммируем только элементы с индексами (i, *, *)
print(np.sum(X, axis=(1, 2)))


# # 5. Линейная алгебра

a = np.array([[2, 1], [2, 3]])
print(a)


# ***Определитель.***


np.linalg.det(a)


# ***Нахождениия обратной.***

b = np.linalg.inv(a)
print(b)



print(a.dot(b))
print(b.dot(a))


# ***Решение НЛУ.***
# $$ A \cdot x = v $$



v = np.array([5, -10])
print(np.linalg.solve(a, v))
print(b.dot(v))


# ***Найдем собственные вектора матрицы A.***
# $$ A \cdot x = \lambda \cdot x $$


l, u = np.linalg.eig(a)
print(l)
print(u)


# ***Собственные значения матриц A и A.T совпадают.***


l, u = np.linalg.eig(a.T)
print(l)
print(u)



l, u = np.linalg.eig(np.eye(3))
print(l)
print(u)


# ***Производительность.***



get_ipython().run_cell_magic('time', '', '\nsum_value = np.sum(np.arange(10**7))\nprint(sum_value)')



get_ipython().run_cell_magic('time', '', 'arr = 5*np.arange(10**7)')

