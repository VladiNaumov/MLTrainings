"""
В этом практическом задании вам предстоит реализовать нейронную сеть,
состоящую из трех нейронов и решающую задачу схожую с описанной в лекции.

Мы возьмем за основу класс Neuron, который вы реализовывали ранее,
и научимся комбинировать выходы отдельных нейронов для получения более сложной разделяющей границы.
"""


# класс Neuron из прошлой практики
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Neuron:
    """
    Класс, реализующий нейрон
    """

    def __init__(self, a: float, b: float, c: float, prob_output: bool = True):
        """
        a,b,c -- коэффиценты (веса) нейрона
        prob_output -- если True, то, на выходе -- вероятности, если False -- логит

        """
        self.a = a
        self.b = b
        self.c = c
        self.prob_output = prob_output

    def calculate_logit(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов.
             Первый столбец -- признак  x1, второй -- x2.

        Данный метод должен возвращать logit = a*x1 + b*x2 + c

        """
        assert np.ndim(x) == 2 and x.shape[1] == 2
        logit = None
        logit = self.a * x[:, 0] + self.b * x[:, 1] + self.c
        return logit

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов.
             Первый столбец -- признак  x1, второй -- x2.

        Данный метод должен возвращать logit(x), если self.prob_output=False,
        и sigmoid(logit(x)) иначе

        """
        assert np.ndim(x) == 2 and x.shape[1] == 2
        logit = self.calculate_logit(x)

        if self.prob_output:
            output = sigmoid(logit)
        else:
            output = logit
        assert output.shape == (
        x.shape[0],), f"Output size must have following shape: {[x.shape[0], ]}. Recieved: {output.shape}"
        return output

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов.
             Первый столбец -- признак  x1, второй -- x2.

        Данный метод должен возвращать предсказанный класс для
        каждого из N объектов -- 0 или 1.

        """
        logit = self.calculate_logit(x)
        predicted_classes = (logit > 0.0).astype(np.int32)

        assert predicted_classes.shape == (
        x.shape[0],), f"Output size must have following shape: {[x.shape[0], ]}. Recieved: {output.shape}"
        return predicted_classes

    def __repr__(self):
        return f"Neuron description. Weights: a={self.a}, b={self.b}. Bias: c={self.c}."

""" Задание 1 Реализуйте метод calculate_logit класса ThreeNeuronsNeuralNet """

from typing import List


class ThreeNeuronsNeuralNet(Neuron):
    """
    Нейронная сеть из трех нейронов.

    """

    def __init__(self, first_neuron_params: List[float],
                 second_neuron_params: List[float],
                 third_neuron_params: List[float]):
        """
        Для конструировани нейронной сети нам потребуются параметры трех нейронов,
        которые передаются в трех списках.

        Мы наследуемся от класса Neuron, т.к. нам нужно переопределить только
        пересчет логитов. Предсказания классов и вероятностей уже реализованы.

        """
        self.prob_output = True  # фиксируем вероятностный выход
        self.neuron1 = Neuron(*first_neuron_params,
                              prob_output=True)  # конструируем первый нейрон, prob_output=True, это важно!
        self.neuron2 = Neuron(*second_neuron_params,
                              prob_output=True)  # конструируем второй нейрон, prob_output=True, это важно!
        self.neuron3 = Neuron(*third_neuron_params, prob_output=self.prob_output)  # конструируем третий нейрон

    def calculate_logit(self, x):
        """
        x -- массив размера (N, 2), где N -- количество объектов.
             Первый столбец -- признак  x1, второй -- x2.
             Важно! Это исходные координаты!

        Этот метод должен вернуть логит предсказанный всей сетью
        Это можно сделать в 4 шага:
        1) Получить вероятности синего класса для исходных данных первым
           нейроном: вектор длины N -- z1
        2) Получить вероятности синего класса для исходных данных вторым
           нейроном: вектор длины N -- z2
        3) Склеить полученные вероятности: массив размера (N, 2) -- z1z2
           * вам может быть полезна функция np.vstack
        4) Получить логит(!, calculate_logit) третьего нейрона, примененного к z1z2 -- logit

        """
        z1 = None
        z2 = None
        z1z2 = None
        logit = None

        # < YOUR CODE STARTS HERE >

        # < YOUR CODE ENDS HERE >

        return logit

""" TEST """

test_net = ThreeNeuronsNeuralNet([1, 2, 3], [4, 5, 6], [7, 8, 9])
test_input = np.array([[10, 20], [30, -40]])
assert np.allclose(test_net.calculate_logit(test_input), np.array([24., 9.]))
print("Simple test passed")


""" Теперь посмотрим как выглядит разделяющая поверхность """

a1, b1, c1 = 0.0, -3.0, 15.0
a2, b2, c2 = -3.0, 3.0, -2.0
a3, b3, c3 = 1.5, 1.5, -0.65

neural_net = ThreeNeuronsNeuralNet([a1, b1, c1],
                                   [a2, b2, c2],
                                   [a3, b3, c3])

X = np.array([[10, 6], [6, 6], [9, 8], [10, 10],
              [10, 4], [4, 4], [4, 6], [8, 9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

visualize(neural_net, X, y)

if eval_clf(neural_net, X, y) == 1:
    print("Well done")
else:
    print("Something went wrong")


""" Задание 2 (бонусное) Подберите коэффициенты таким образом, 
чтобы идеально разделить данные ниже. Совет: настраивайте параметры по-очереди, начиная с первого нейрона. """

np.random.seed(10)
X = [[1.5  + np.random.rand() - 0.5, x] for x in np.arange(10)]
X += [[8.5  + np.random.rand() - 0.5, x] for x in np.arange(10)]
X += [[5  + np.random.rand() - 0.5, x] for x in np.arange(10)]
X = np.array(X)
y = np.zeros(len(X))
y[-10:] = 1

show_data(X, y)

################################
# Меняйте коэффиценты ниже
a1, b1, c1 = 0.0, -3.0, 15.0
a2, b2, c2 = -3.0, 3.0, -2.0
a3, b3, c3 = 1.5, 1.5, -0.65

################################
neural_net = ThreeNeuronsNeuralNet([a1, b1, c1],
                                   [a2, b2, c2],
                                   [a3, b3, c3])

visualize(neural_net, X, y)

if eval_clf(neural_net, X, y) == 1:
    print("Well done")
else:
    print("Something went wrong")




