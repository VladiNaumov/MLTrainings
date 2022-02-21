import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

data = pd.read_csv("../data/Advertising.csv")

# Чтобы увидеть, как выглядят данные, мы делаем следующее:
data.head()

print(data.columns)

""" OUT 
Index(['Unnamed: 0', 'TV', 'radio', 'newspaper', 'sales'], dtype='object') 

"""

data.drop(['Unnamed: 0'], axis=1)


# ## Simple linear regression 

# Мы используемmatplotlib, популярная библиотека для построения графиков на Python

plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black'
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()


X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)


print(reg.coef_[0][0])
print(reg.intercept_[0])

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

# Давайте представим, как линия соответствует данным.

predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black'
)
plt.plot(
    data['TV'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()

# Оценка актуальности модели

X = data['TV']
y = data['sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


### Multiple linear regression

Xs = data.drop(['sales', 'Unnamed: 0'], axis=1)
y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(Xs, y)
print(reg.coef_)
print(reg.intercept_)
print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))

reg.score(Xs, y)

X = np.column_stack((data['TV'], data['radio'], data['newspaper']))
y = data['sales']

# Оценка актуальности модели
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
