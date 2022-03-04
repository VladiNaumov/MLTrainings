import numpy as np


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

        # переменная logit должна содержать логит, как описано в комментарии к функции
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

        # < YOUR CODE STARTS HERE >
        # переменная output должна содежать logit(x), если self.prob_output = False
        # и sigmoid(logit(x)) иначе
        logit = self.calculate_logit(x)

        if self.prob_output:
            output = sigmoid(logit)
        else:
            output = logit
        # < YOUR CODE ENDS HERE >

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
        # < YOUR CODE STARTS HERE >
        # переменная predicted_classes должна содержать предсказынные классы для всех объектов
        # не забывайте про уже реализованные функции )

        logit = self.calculate_logit(x)
        predicted_classes = (logit > 0.0).astype(np.int32)
        # < YOUR CODE ENDS HERE >

        assert predicted_classes.shape == (
        x.shape[0],), f"Output size must have following shape: {[x.shape[0], ]}. Recieved: {output.shape}"
        return predicted_classes

    def __repr__(self):
        return f"Neuron description. Weights: a={self.a}, b={self.b}. Bias: c={self.c}."


def sigmoid(x):

    # value должна содержать выход сигмоиды
    value = 1. / (1. + np.exp(-x))
    return value
