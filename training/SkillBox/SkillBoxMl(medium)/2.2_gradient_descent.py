""" Задание 2 Продемонстрируйте на примере функции:

1. застревание в локальном минимуме (выберете другое начальное приближение)
2. расхождение оптимизации (попробуйте другую скорость обучения) """

from training.SkillBoxNeuralNetworks.gradient_descent import gradient_descent, visualize

# Задание 2.1

# градиентный спуск


initial_position = 3
f = lambda x: (x**2 - 4*x + 4)*(x**2 + 4*x + 2)

# Напишите, чему равна производная f по x
df_dx = lambda x: 4*(x**3 - 5*x + 2)

# Выберите подходящие параметры для оптимизации
n_iters =  30
lr = 5e-3


position, positions, values = gradient_descent(f=f, df_dx=df_dx, initial_position=initial_position,
                                               n_iters=n_iters, lr=lr, tol=1e-4)
visualize(f, (-4, 4), positions=positions, values=values)

# Задание 2.1

# градиентный спуск
initial_position = 0
f = lambda x: (x**2 - 4*x + 4)*(x**2 + 4*x + 2)


# Напишите, чему равна производная f по x
df_dx = lambda x: 4*(x**3 - 5*x + 2)



# Выберите подходящие параметры для оптимизации
n_iters =  3
lr = 5e-1



position, positions, values = gradient_descent(f=f, df_dx=df_dx, initial_position=initial_position,
                                               n_iters=n_iters, lr=lr, tol=1e-4)
visualize(f, (-4, 4), positions=positions, values=values)