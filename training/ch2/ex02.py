"""
  методы ML обучения с учителем:
  Ближайшие соседи
  Подходит для небольших наборов данных, хорош в качестве базовой   модели, прост в объяснении.

"""
# KNeighborsClassifier()
# В k Классификация соседей KNeighborsClassifier — наиболее часто используемый метод.
# Оптимальный выбор стоимостиk сильно зависит от данных: как правило, более крупный k подавляет влияние шума,
# но делает границы классификации менее четкими.

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier


# генерируем набор данных
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 3, figsize=(10, 3))


for n_neighbors, ax in zip([1, 3, 9], axes):
    # создаем объект-классификатор и подгоняем в одной строке
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("количество соседей:{}".format(n_neighbors))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
plt.show()


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

# пробуем n_neighbors от 1 до 10
neighbors_settings = range(1, 11)



for n_neighbors in neighbors_settings:
    # строим модель
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # записываем правильность на обучающем наборе
    training_accuracy.append(clf.score(X_train, y_train))
    # записываем правильность на тестовом наборе
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()
plt.show()

# n_neighbors=1 -mустанавливаем одного ближайшего соседа
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

# n_neighbors=1 -mустанавливаем одного ближайшего соседа
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()



