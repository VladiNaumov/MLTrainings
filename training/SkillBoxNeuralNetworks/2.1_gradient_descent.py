"""
Градиентный спуск
В этом практическом занятии вы реализуете градиентный спуск самостоятельно.
На сегодняшний день -- это основной способ оптимизации нейронных сетей.

Мы начнем с простой функции -- парабола, затем перейдем к чуть более сложной.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f: callable, df_dx: callable, initial_position: float, n_iters: int, lr: float,
                     tol: float = 1e-3):
    """
    Алгоритм градиентного спуска.

    f -- функция, минимум которой мы хотим найти
    df_dx -- производная функции по x
    initial_position -- начальное приближение
    n_iters -- максимальное количество шагов градиентного спуска
    lr -- learning rate, скорость обучения
    tol -- точность с которой мы будем считать, что значение функции не меняется

    Функция должна вернуть найденный минимум, историю положений  и значений функции
    """
    positions = []
    values = []
    position = initial_position
    for i in range(n_iters):
        positions.append(position)
        values.append(f(position))
        if len(values) > 1:
            if np.abs(values[-1] - values[-2]) < tol:
                break

        # Реализуйте шаг градиентного спуска.
        position = position - lr * df_dx(position)

    print(f"Found minimum at x={position} after {len(positions) - 1} steps.")
    return position, positions, values


def visualize(f: callable, limits: tuple, positions: list, values: list):
    xs = np.linspace(limits[0], limits[1], 100)
    plt.figure(figsize=(10, 5))
    plt.plot(xs, f(xs))
    plt.title(f"Найденный минимум после {len(positions) - 1} шагов: $f = {values[-1]:.2}$ при $x = {positions[-1]:.2}$")
    _, = plt.plot(positions, values)
    for p, v in zip(positions, values):
        plt.scatter(p, v, s=30, zorder=4)


# градиентный спуск

f = lambda x: x**2 + x + 1 # функция для оптимизации
df_dx = lambda x: 2*x + 1 # производная

n_iters = 20
lr = 0.9

initial_position = -4.
position, positions, values = gradient_descent(f=f, df_dx=df_dx, initial_position=initial_position,
                                               n_iters=n_iters, lr=lr, tol=1e-4)

visualize(f, (-4, 3), positions=positions, values=values)

plt.show()

# градиентный спуск 

initial_position = 4
f = lambda x: (x**2 - 4*x + 4)*(x**2 + 4*x + 2)
df_dx = lambda x: 4*(x**3 - 5*x + 2)


# Выберите подходящие параметры для оптимизации
n_iters = 30
lr = 2e-2


position, positions, values = gradient_descent(f=f, df_dx=df_dx, initial_position=initial_position,
                                               n_iters=n_iters, lr=lr, tol=1e-4)
visualize(f, (-4, 4), positions=positions, values=values)

plt.show()


