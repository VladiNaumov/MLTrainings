import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    # реализуйте сигмоиду 1./(1 + exp(-x))
    value = None

   # value должна содержать выход сигмоиды
    value = 1. / (1. + np.exp(-x))
    return value


plt.figure(figsize=(10, 5))
plt.title("$\sigma(x)$")
xs = np.linspace(-5, 5, 100)
_ = plt.plot(xs, sigmoid(xs))

for x, y in zip([-5, 0, 5], [0.006, 0.5, 0.993]):
    assert np.allclose(sigmoid(x), y, atol=1e-3), f"Value at {x} is {sigmoid(x)}, but {y} is expected"
print("Tests passed")
plt.show()