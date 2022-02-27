"""
Деревья решений - один из наиболее популярных методов классификации. Одной из причин их популярности является то, что окончательную модель крайне легко понять - достаточно построить граф решений и посмотреть, почему был сделан тот или иной прогноз.

Также деревья решений являются основой таких методов как бэггинг, случайные леса и градиентный бустинг, о которых будем говорить позднее

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Загружаем набор данных Ирисы:
iris_dataset = load_iris()

# Посмотрим, что включает в себя набор данных.
iris_dataset.keys()

# делаем свои переменные
a = iris_dataset.data[:, 2:4]
d = iris_dataset['target']

"""
Разобьем данные на тренировочный и тестовый датасеты и для простоты реализации алгоритма используя метод train_test_split()
объединим массивы признаков объектов и метки их классов, чтобы было понятно, к какому классу относится каждый объект.
"""
x_train, x_test, y_train, y_test = train_test_split(a, d, random_state=0) # random_state - для воспроизводимости

""" DecisionTreeClassifier — это класс, способный выполнять мультиклассовую классификацию набора данных. """
dtc = DecisionTreeClassifier()

dtc_model = dtc.fit(x_train, y_train)

from sklearn import tree
import graphviz

""" Визуализируем граф обученной модели. Для этого понадобится установить библиотеку graphviz. """
def print_graph(data):
    dot_data = tree.export_graphviz(data, out_file=None, feature_names=iris_dataset.feature_names[2:4],
                                    class_names=iris_dataset.target_names, filled=True)
    return graphviz.Source(dot_data)

print_graph(dtc_model)

""" Получим предсказания для тестовых данных. """
dtc_predictions = dtc.predict(x_test)

""" Для определения точности предсказаний воспользуемся встроенной функцией score. """
accuracy = dtc.score(x_test, y_test)
print(f'Accuracy: {accuracy}')

""" Теперь построим и обучим модель с критерием качества разбиения энтропия """
dtc_entrp = DecisionTreeClassifier(criterion='entropy')
dtc_model_entrp = dtc_entrp.fit(x_train, y_train)
dtc_predictions_entrp = dtc_entrp.predict(x_test)
accuracy = dtc_entrp.score(x_test, y_test)
print(f'Accuracy: {accuracy}')

print_graph(dtc_model_entrp)

"""
Как мы можем видеть, при таких параметрах алгоритм работает менее точно, чем kNN и наивный Байес. 
Можно предположить, что алгоритму не хватает двух признаков для качественного разбиения на поддеревья - 
на большем количестве данных по каждому объекту метод скорее всего будет работать более точно.

Определенно, деревья решений имеют ряд плюсов: они легко интерпретируемы, визуализируемы (исчезает эффект "черного ящика"), 
достаточно быстро обучаются и делают прогнозы, имеют небольшое количество параметров модели и поддерживают как категориальные, 
так и числовые признаки.

Однако при этом они очень чувствительны к шумам во входных данных,
 подвержены переобучению - для борьбы с ним необходимо корректно устанавливать гиперпараметры 
 (максимальную глубину дерева или минимальное число элементов в листьях деревьев), а также не умеют предсказывать данные, 
 выходящие за рамки обучающего датасета
"""