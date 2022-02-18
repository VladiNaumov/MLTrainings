"""
Математическая магия регуляризации
Регуляризация - это способ борьбы с таким явлением как "переобучение".

Мы узнали, как применять регуляризацию для борьбы с переобучением.
Мы уже умеем определять, что модель переобучилась - нужно обучать на тренировочном сете, а метрику качества считать на валидационном.
Давайте построим несколько моделей полиномиальной регрессии для разных степеней полинома,
чтобы выявить "волшебную" степень, в которой недообученная модель становится переобученной

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('../data/non_linear.csv', sep=',')
print(data.head())

# основной график
plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8)
plt.show()


def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
          source_data**n for n in range(1, degree + 1)
    ]).T



def train_polynomial(degree, data):
    """Генерим данные, тренируем модель дополнительно рисуем график """
    X = generate_degrees(data['x_train'], degree)
    y = data.y_train.values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_pred_train = model.predict(X_train)
    error_valid = mean_squared_error(y_valid, y_pred)
    error_train = mean_squared_error(y_train, y_pred_train)
    print(
        "Степень полинома %d\nОшибка на валидации %.3f\nОшибка на обучении %.3f" %
        (degree, error_valid, error_train)
    )
	# порядок на тестовых данных
    order_test = np.argsort(X_valid[:,0])
    print(order_test)
    print(X_valid[:, 0])
    plt.scatter(X_valid[:,0][order_test], y_valid[order_test], 40, 'r', 'o', alpha=0.8)
    print("Норма вектора весов \t||w|| = %.2f" % (norm(model.coef_)))

    # визуализируем решение
    x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)
    y_linspace = model.predict(generate_degrees(x_linspace, degree))
    plt.plot(x_linspace, y_linspace)
    return error_valid, error_train, norm(model.coef_)

# для сохранение результатов
degrees = []
valid_errors = []
train_errors = []
w_norm = []

degree = 3

error_valid, error_train, coef_norm = train_polynomial(degree, data)

degrees.append(degree)
valid_errors.append(error_valid)
train_errors.append(error_train)
w_norm.append(coef_norm)

degree = 5

error_valid, error_train, coef_norm = train_polynomial(degree, data)

degrees.append(degree)
valid_errors.append(error_valid)
train_errors.append(error_train)
w_norm.append(coef_norm)

degree = 10

error_valid, error_train, coef_norm = train_polynomial(degree, data)

degrees.append(degree)
valid_errors.append(error_valid)
train_errors.append(error_train)
w_norm.append(coef_norm)

# создание графика что бы увидеть в каком месте данные становятся переобученными
fig, ax = plt.subplots()
ax.plot(degrees, valid_errors, 'k--', label='Validation error')
ax.plot(degrees, train_errors, 'k:', label='Train error')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()



#plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')

# модель с регулерезацией
model_ridge = Ridge(alpha=0.01)
# модель без регулерезации
model_linear = Ridge(alpha=0.0)
degree = 10

X = generate_degrees(data['x_train'], degree)
y = data['y_train']
# обучаем модель
model_ridge.fit(X, y)
model_linear.fit(X, y)

x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)

# предсказание линейное регрессии
y_linspace_linear = model_linear.predict(generate_degrees(x_linspace, degree))
# предсказание для ридж регрессии 
y_linspace_ridge = model_ridge.predict(generate_degrees(x_linspace, degree))

# построение графиков
plt.plot(x_linspace, y_linspace_linear)
plt.plot(x_linspace, y_linspace_ridge)

plt.show()


print("Норма вектора весов Ridge \t||w|| = %.2f" % (norm(model_ridge.coef_)))
print("Норма вектора весов Linear \t||w|| = %.2f" % (norm(model_linear.coef_)))



""" подбор правельного значение регулерезации для исключения случаев переобучения """

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

# значение регулерезации для исключения случаев переобучения
alphas = [0.1, 0.15, 0.35 ,0.5,0.8]

best_alpha = alphas[0]
best_rmse = np.infty

for alpha in alphas:
    model_ridge = Ridge(alpha=alpha)
    # обучаем линейную регрессию с  регуляризацией
    model_ridge.fit(X_train, y_train)
    y_pred = model_ridge.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    if error < best_rmse:
        best_rmse = error
        best_alpha = alpha
    print("alpha =%.2f Ошибка %.5f" % (alpha, error))
print('\n-------\nЛучшая модель aplpha=%.2f с ошибкой RMSE=%.5f\n-------' % (best_alpha, best_rmse))

"""
В этом уроке мы научились делать наши модели намного стабильнее - теперь веса линейной регрессии не увеличиваются с ростом степени полинома, 
а остаются заключёнными в допустимых пределах. Мы научились уменьшать переобучение с помощью регуляризации модели.

Однако, мы столкнулись с ворненгом LinAlgWarning что это за ошибка? 
Почему она возникает? Как её побороть? Об этом мы поговорим в следующем уроке.
"""