# пользоваться нейроном можно следующим образом:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from sklearn.metrics import accuracy_score

from training.SkillBox.SkillBoxNeuralNetworks.Neuron import Neuron

""" вспомогательный код """

np.random.seed(10)
colors = ['red', "blue"]
labels_cmap = ListedColormap(colors, 2)
colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # R -> W -> B
main_cmap = LinearSegmentedColormap.from_list("main_scheme", colors, N=300)


def show_data(X, y):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], s=120, color=labels_cmap(y))


def generate_simple_data(N, a, b, c, max_x=5, max_y=5):
    np.random.seed(10)
    X = np.random.rand(N, 2)
    X[:, 0] = X[:, 0] * max_x
    X[:, 1] = X[:, 1] * max_y
    y = np.zeros(N)
    y[X[:, 0] * a + X[:, 1] * b + c > 0] = 1
    # y[X[:, 0] * a + X[:, 1] * b + c < -m]= 0
    return X, y


def show_descision_boundary(clf, limits, binary=False, X=None, y=None, n_lines=10, show_lines=False,
                            figsize=(5, 5), ax=None):
    xs, ys = limits
    x_min, x_max = xs
    y_min, y_max = ys

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

    if binary:
        Z = clf.predict_class(np.c_[xx.ravel(), yy.ravel()])
        norm = Normalize(vmin=0., vmax=1.)
    else:
        Z = clf(np.c_[xx.ravel(), yy.ravel()])
        if clf.prob_output:
            norm = Normalize(vmin=0., vmax=1.)
        else:
            norm = Normalize(vmin=-10., vmax=10., clip=True)

    Z = Z.reshape(xx.shape)
    Z = Z.astype(np.float32)

    ax.contourf(xx, yy, Z, n_lines, alpha=0.4, cmap=main_cmap, norm=norm)
    if show_lines:
        cp = ax.contour(xx, yy, Z, n_lines)
        ax.clabel(cp, inline=True,
                  fontsize=10, colors="green")

    if y is not None:
        X = np.array(X)
        y = np.array(y)
        ax.scatter(X[:, 0], X[:, 1], s=120, color=labels_cmap(y),
                   zorder=4)


def eval_clf(clf, X, y):
    acc = accuracy_score(clf.predict_class(X), y)
    print(f"Accuracy {acc}")
    return acc


def create_three_axes():
    fig = plt.figure(figsize=(16, 5))
    grid = plt.GridSpec(1, 3)
    logits = fig.add_subplot(grid[0, 0])
    logits.set_xlabel("$x_1$")
    logits.set_ylabel("$x_2$")
    logits.set_title("Логит класса 1 (синий)")

    probs = fig.add_subplot(grid[0, 1])
    probs.set_xlabel("$x_1$")
    probs.set_ylabel("$x_2$")
    probs.set_title("Вероятность класса 1 (синий)")

    binary = fig.add_subplot(grid[0, 2])
    binary.set_xlabel("$x_1$")
    binary.set_ylabel("$x_2$")
    binary.set_title("Решающая поверхность")

    return logits, probs, binary

""" END """


#  пользоваться нейроном можно следующим образом:
x_random = np.random.rand(5, 2) # случайные данные

neuron_sigmoid = Neuron(1, -1, 0, prob_output=True) # нейрон, на выходе которого вероятность
neuron_logit = Neuron(1, -1, 0, prob_output=False) # нейрон, на выходе которого логит

probabilities = neuron_sigmoid(x_random)
logits = neuron_logit(x_random)
print(f"Neuron_logit: {neuron_logit}") # выведем описание нейрона
print(f"Logits: {logits}") # предсказанные логиты
print(f"Probabilities: {probabilities}") # предсказанные вероятности
print(f"Classes: {neuron_logit.predict_class(x_random)}") # предсказанные классы

# два нейрона имеют одинаковые параметры, значит предсказанные классы должны совпадать
classes_1 = neuron_logit.predict_class(x_random)
classes_2 = neuron_sigmoid.predict_class(x_random)
assert np.alltrue( classes_1 == classes_2)



n_lines = 10
limits = [[-1, 11], [-1, 11]]
for i, params in enumerate([[1, -1, 0], [2, -0.5, -4], [-1, 2, -10]]):
    a, b, c = params
    clf_logits = Neuron(a, b, c, False)
    clf_probs = Neuron(a, b, c, True)

    X, y = generate_simple_data(20, a, b, c, 10, 10)

    logits, probs, classes = create_three_axes()

    show_descision_boundary(limits=limits, clf=clf_logits,
                            X=X,
                            y=y,
                            n_lines=n_lines,
                            show_lines=True, ax=logits)

    show_descision_boundary(limits=limits, clf=clf_probs,
                            X=X,
                            y=y,
                            n_lines=n_lines,
                            show_lines=True, ax=probs)

    show_descision_boundary(limits=limits, clf=clf_logits, binary=True,
                            X=X,
                            y=y,
                            n_lines=n_lines,
                            show_lines=False, ax=classes)

    acc = eval_clf(clf_logits, X, y)