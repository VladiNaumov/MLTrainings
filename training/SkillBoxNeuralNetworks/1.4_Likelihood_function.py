"""Задание 1 Реализуйте функции, необходимые для вычисления правдоподобия и логарифма правдоподобия.
"""

import numpy as np
import matplotlib.pyplot as plt


def p_hat(p, y):
    # p - предсказанная моделью вероятность класса 1
    # y - реальный класс (0 или 1)

    # функция должна вернуть вероятность класса y, при предсказании
    # модели p

    # < YOUR CODE STARTS HERE >
    if y == 1:
        return p
    else:
        return (1 - p)
    # < YOUR CODE ENDS HERE >


def log_p_hat(p, y):
    return np.log(p_hat(p, y))


def likelihood(ps, ys):
    # ps - предсказанные вероятности класса 1 моделю для N объектов
    # ys - реальные классы N объектов

    # функция должна использовать p_hat и возвращать правдоподобие

    likelihood_ = None
    # < YOUR CODE STARTS HERE >
    probs = [p_hat(p, y) for (p, y) in zip(ps, ys)]
    likelihood_ = np.prod(probs)
    # < YOUR CODE ENDS HERE >
    return likelihood_


def loglikelihood(ps, ys):
    # ps - предсказанные вероятности класса 1 моделю для N объектов
    # ys - реальные классы N объектов

    # функция должна использовать log_p_hat и возвращать логарифм правдоподобия
    # (на количество делить не нужно)

    p = np.clip(ps, a_min=1e-6, a_max=1 - 1e-6)
    log_probs = [log_p_hat(p, y) for (p, y) in zip(ps, ys)]
    loglikelihood_ = None
    # < YOUR CODE STARTS HERE >
    loglikelihood_ = np.sum(log_probs)
    # < YOUR CODE ENDS HERE >
    return loglikelihood_


test_ps = [0.1, 0.2, 0.3, 0.4]
test_ys = [0, 1, 0, 1]
assert likelihood(test_ps, test_ys) == 0.0504
assert np.allclose(np.log(likelihood(test_ps, test_ys)), loglikelihood(test_ps, test_ys))
print("Tests passed!")


"""  Применение принципа максимума правдоподобия на практике.

Предположим у нас имеется монетка, подбросив которую 10 раз мы получили следующие результаты (0 -- решка, 1 -- орел): y=[1,0,1,1,1,1,0,0,0,1] 

Наша задача узнать "параметры" этой монетки, а именно вероятность выпадения орла (p). 
Интуитивно понятно, что  p=0.6 , но давайте теперь получим это значение с помощью принципа максимума правдоподобия.
Ваша задача используя код ниже показать что правдоподобие и его логарифм достигают экстремумов в 0.6. Мы будем это делать "в лоб". А именно мы посчитаем их значения для всех возможных вероятностей (с шагом 0.01) и визуально оценим их.
Подсказка: монетка это модель, которая не зависит от объекта и имеет только один параметр -- с какой вероятностью она выпадает орлом. """

coins = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 1])

N = 100
p_to_try =  np.linspace(0, 1, N)

# < YOUR CODE STARTS HERE >
# Переменная liks должна содержать значения правдоподобия для каждой из 100 вероятностей в p_to_try.
# Переменная logliks должна содержать значения логарифма правдоподобия для каждой из 100 вероятностей в p_to_try.

liks =
logliks =

# < YOUR CODE ENDS HERE >

max_lik_ind = np.argmax(liks)
max_loglik_ind = np.argmax(logliks)

fig = plt.figure(figsize=(15, 5))
grid = plt.GridSpec(1, 2)
plt.subplot(grid[0, 0])
plt.plot(p_to_try, liks)
plt.xlabel("$p$")
plt.title(f"Likelihood. Max at $p={p_to_try[max_lik_ind]:.2}$")
plt.scatter(p_to_try[max_lik_ind], liks[max_lik_ind], c='red')

plt.subplot(grid[0, 1])
plt.plot(p_to_try, -np.array(logliks))
plt.scatter(p_to_try[max_loglik_ind], -logliks[max_loglik_ind], c='red')
plt.title(f"Negative Log Likelihood. Min at $p={p_to_try[max_loglik_ind]:.2}$")
_ = plt.xlabel("$p$")