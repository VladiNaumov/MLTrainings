import numpy as np
import matplotlib.pyplot as plt


def p_hat(p, y):
    # p - предсказанная моделью вероятность класса 1
    # y - реальный класс (0 или 1)

    # функция должна вернуть вероятность класса y, при предсказании
    # модели p

    if y == 1:
        return p
    else:
        return (1 - p)


# вычисление логарифма p с крышкой
def log_p_hat(p, y):
    print(np.log(p_hat(p, y)))
    return np.log(p_hat(p, y))

print(p_hat(1, 1))
print(log_p_hat(10, 1))