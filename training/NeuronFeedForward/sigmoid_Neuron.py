import numpy as np

""" Простой пример работы с нейронами в Python

Предположим, у нас есть нейрон с двумя входами, который использует функцию активации сигмоида и имеет следующие параметры:
weights = [0,1], bias = 4

формулы функции сигмоида
w = [0,1] — это просто один из способов написания w1 = 0, w2 = 1 в векторной форме.
 Присвоим нейрону вход со значением x = [2, 3]. 
 Для более компактного представления будет использовано скалярное произведение. """

def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Вводные данные о весе, добавление смещения и последующее использование функции активации

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


""" нейрон с двумя входами weights[0, 1] """
weights = np.array([0, 1])

""" bias смещение """
bias = 4

n = Neuron(weights, bias)

x = np.array([2, 3])  # x1 = 2, x2 = 3
print(n.feedforward(x))  # 0.9990889488055994