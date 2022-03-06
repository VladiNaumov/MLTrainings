#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np


# In[7]:


def act(x):
    return 0 if x < 0.5 else 1

def go(R1 = 0, R2 = 0, R3 = 1 ):

    x = np.array([R1, R2, R3])

    # Вес для первого нейрона скрытого слоя 'W1'
    w11 = [0.3, 0.3, 1]

    # Вес для второго нейрона скрытого слоя 'W2'
    w12 = [0.4, -0.5, 0]

    # Вес для второго нейрона скрытого слоя 'W2'
    w13 = [0.3, 0.3, 1]

    w14 = [0.3, 0.3, 0]

    weight1 = np.array([w11, w12, w13, w14])  # матрица 3x3
    weight2 = np.array([1, 0, 0, 1])  # вектор 1х3

    # sum = R1W1 + R2W2 + R3W3
    sum_hidden = np.dot(weight1, x)  # вычисляем сумму на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    # выходной нейрон
    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    # вычисляем сумму
    sum_end = np.dot(weight2, out_hidden)
    print("вычисляем Y " + str(sum_end))

    y = act(sum_end)

    print("Выходное значение НС: " + str(y))

    return y


go()

