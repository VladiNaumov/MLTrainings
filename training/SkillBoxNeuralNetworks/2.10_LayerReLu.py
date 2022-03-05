"""
Обучение нейронной сети
Это последняя и самая важная практика в этом блоке. В ней вы соберете воедино все, что мы с вами изучили и примените для создания сети, которая классифицирует рукописные цифры.

Задание будет состоять из следующих этапов:

1.Реализация слоя ReLU
2.Реализация полносвязного слоя
3.Написание обучающего цикла
4.Загрузка данных и обучение сети
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

class Layer:
    """
    Базовый класс слоя нашей нейронной сети.
    Все слои должны наследоваться от него и реализовывать два метода: forward и backward
    """

    def forward(self, x):
        pass

    def backward(self, dL_dz, learning_rate=0):
        pass


class ReLU(Layer):
    """
    Слой ReLU
    """

    def forward(self, x):
        """
        Метод, который вычисляет ReLU(x)

        Размер выхода должен совпадать со входом

        """

        self._saved_input = x  # нам необходимо сохранить вход
        output = None

        # < YOUR CODE STARTS HERE >
        # переменная output должна содержать выход ReLU

        # подсказка: метод clip библиотеки numpy позволяет при заданном интервале значения вне интервала обрезать по краям интервала.
        # Например, если указан интервал [0, 1],
        # значения меньше 0 становятся 0, а значения больше 1 становятся 1.

        output =

        # < YOUR CODE ENDS HERE >
        assert output.shape == x.shape
        return output

    def backward(self, dL_dz, learning_rate=0.):
        """
        dL_dz -- производная финальной функции по выходу этого слоя.
                 Размерость должна в точности соответствовать размерности
                 x, который прошел в forward pass.
        learning_rate -- не используется, т.к. ReLU не содержит параметров.

        Метод должен посчитать производную dL_dx.
        Благодаря chain rule, мы знаем, что dL_dx = dL_dz * dz_dx
        и при этом dL_dz нам известна.

        Для слоя relu, dz_dx(x) = 1, при x > 0, и dz_dz = 0 при x < 0

        * сохраненный инпут находится в self._saved_input
        """
        dz_dx = None

        # < YOUR CODE STARTS HERE >
        # переменная dz_dx должна содержать производную выхода ReLU по ее входу

        dz_dx =

        # < YOUR CODE ENDS HERE >
        assert dz_dx.shape == self._saved_input.shape, f"Shapes must be the same. Got {dz_dx.shape, self._saved_input.shape}"
        output = dz_dx * dL_dz
        return output

relu = ReLU()

# убедитесь, что график соответствует представленному вверху
plt.plot(np.linspace(-1, 1, 100), relu.forward(np.linspace(-1, 1, 100)))

f = lambda x: ReLU().forward(x)

x = np.linspace(-1, 1, 10*32).reshape([10, 32])
l = ReLU()
l.forward(x)
grads = l.backward(np.ones([10, 32]))
numeric_grads = derivative(f, x, dx=1e-6)
assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0),\
     "gradient returned by your layer does not match the numerically computed gradient"
print("Test passed")

""" Задание 2 Реализация полносвязного слоя. """

class FCLayer(Layer):
    """
    Полносвязный (fully connected/dense) слой
    """

    def __init__(self, in_dim, out_dim):
        """
        in_dim, out_dim -- количество входных и выходных нейронов соответственно
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        # инициализируем матрицу весов (in_dim,out_dim) нормальным распределением
        self.weights = np.random.randn(in_dim, out_dim) * 0.001

        # инициализируем смещение нулями
        self.bias = np.zeros(self.out_dim)
        self._saved_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисление выхода полносвязного слоя.

        x -- вход слоя, размерности (N, in_dim), где N -- количество объектов
             в батче

        return: matmul(x, weights) + bias
        """
        assert np.ndim(x) == 2
        assert x.shape[1] == self.in_dim
        self._saved_input = x

        # < YOUR CODE STARTS HERE >
        # переменная output должна содержать выход полносвязного слоя

        output =

        # < YOUR CODE ENDS HERE >

        assert output.shape == (x.shape[0], self.out_dim), (output.shape, (x.shape[0], self.out_dim))
        return output

    def backward(self, dL_dz, learning_rate=0.):
        """
        dL_dz -- производная финальной функции по выходу этого слоя.
                 Размерость (N, self.out_dim).
        learning_rate -- если отличен от нуля, то с вызовом этой функции, параметры
                         слоя (weights, bias) будут обновлены

        Метод должен посчитать производную dL_dx.

        """
        assert np.ndim(dL_dz) == 2
        assert dL_dz.shape[1] == self.out_dim

        # очень рекомендуем понять почему это так,
        # хорошее объяснение здесь: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        self.dL_dw = np.dot(self._saved_input.T, dL_dz)
        self.dL_dx = np.dot(dL_dz, self.weights.T)
        self.dL_db = dL_dz.sum(0)

        assert self.dL_db.shape == self.bias.shape
        assert self.dL_dw.shape == self.weights.shape
        assert self.dL_dx.shape == self._saved_input.shape

        if learning_rate != 0:
            # знакомый вам шаг градиентного спуска!
            self.weights -= learning_rate * self.dL_dw
            self.bias -= learning_rate * self.dL_db

        return self.dL_dx



