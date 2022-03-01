
"""
Постановка ML задачи кластеризации
Кластерный анализ (Data clustering) — задача разбиения заданной выборки объектов на непересекающиеся подмножества - такие множества называются кластерами.
Каждый кластер должен состоять из схожих объектов, а объекты разных кластеров должны существенно отличаться.
Это задача обучения без учителя - истинных меток классов алгоритм не требует.
"""

from scipy.spatial.distance import euclidean

c1 = [1.0, 1.5]
c2 = [-1.0, -0.5]

dist = euclidean(c1, c2)
print("Расстояние между кластерами c1 и c2: %.4f" % dist)


""" Алгоритм K-средних (K-means) """

# Для иллюстрации работы алгоритма, загрузим датасет для кластеризации
import pickle
import numpy as np

# данные получены с помощью функции make_classification
with open('../../data/clustering.pkl', 'rb') as f:
    data_clustering = pickle.load(f)

X = np.array(data_clustering['X'])
Y = np.array(data_clustering['Y'])


# Визуализируем датасет, раскрасив два кластера в разные цвета
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, marker='o', alpha=0.8, label='data')
plt.show()

""" В библиотеке sklearn уже есть готовая реализация алгоритма sklearn.cluster.KMeans - давайте просто применим её к нашим данным. 
Точки разобъём на два кластера (параметр n_clusters=2): """

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X)

plt.scatter(X[:, 0], X[:, 1], s=40, c=kmeans_model.labels_, marker='o', alpha=0.8, label='data')
plt.show()

