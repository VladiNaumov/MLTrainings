import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(house, rock, attr):

    x = np.array([house, rock, attr])

# Вес для первого нейрона скрытого слоя 'W1'
    w11 = [0.3, 0.3, 0]

# Вес для второго нейрона скрытого слоя 'W2'
    w12 = [0.4, -0.5, 1]

    weight1 = np.array([w11, w12])  # матрица 2x3
    weight2 = np.array([-1, 1])     # вектор 1х2

# |W1X1| [0.3, 0.3, 0]
# |W2X2| [0.4,-0.5, 1] * | X | 1, 0, 1
    sum_hidden = np.dot(weight1, x)       # вычисляем сумму на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))

# выходной нейрон
    out_hidden = np.array([act(a) for a in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))

# вычисляем сумму weight2 out_hidden
    sum_end = np.dot(weight2, out_hidden)

    y = act(sum_end)

    print("Выходное значение НС: "+str(y))

    return y

# предподчения или входные данные с нейрона '1' да, '0' нет
house = 1
rock = 0
attr = 1

res = go(house, rock, attr)

 
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")
