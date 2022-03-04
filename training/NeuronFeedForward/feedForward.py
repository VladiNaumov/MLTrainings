"""
Создание нейронной сети прямое распространение FeedForward

Пример сбор нейронов в нейросеть
Нейронная сеть по сути представляет собой группу связанных между собой нейронов

"""

import numpy as np

from training.NeuronFeedForward.sigmoid_Neuron import Neuron


class OurNeuralNetwork:
    """
    Нейронная сеть, у которой:
        - 2 входа
        - 1 скрытый слой с двумя нейронами (h1, h2)
        - слой вывода с одним нейроном (o1)
    У каждого нейрона одинаковые вес и смещение:
        - w = [0, 1]
        - b = 0
    """

    def __init__(self):
        """ weights - 2 входа """
        weights = np.array([0, 1])
        """ смещение """
        bias = 0

        # Класс Neuron из предыдущего раздела
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Вводы для о1 являются выводами h1 и h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421

y = np.array([1, 3])
print(network.feedforward(y))  # 0.7216325609518421