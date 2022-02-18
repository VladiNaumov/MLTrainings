Book introduction_to_ml_with_python

(https://mybinder.org/v2/gh/amueller/introduction_to_ml_with_python/master)

# Machine Learning with Python

training/ch1/ex01 -> работа с данными 
training/ch1/ex02 -> Теперь мы можем начать строить реальную модель машинного обучения (В даном примере мы будем использовать модель обучния "'k' ближайших  соседей" )

training/ch2/ -> методы ML обучения с учителем:

мы рассмотрели широкий спектр моделей машинного обучения для  классификации  и  регрессии,  их  преимущества  и  недостатки, 
настройки  сложности  для  каждой  модели.

Ниже  дается  краткий  обзор  случаев  использования  той  или иной модели:

Ближайшие соседи (training/ch2/ -> ex01, ex02, ex03)
Подходит для небольших наборов данных, хорош в качестве базовой 
модели, прост в объяснении. 
 
Линейные модели (training/ch2/ -> ex04)
Считается первым алгоритмом, который нужно попробовать, хорош 
для  очень  больших  наборов  данных,  подходит  для  данных  с  очень 
высокой размерностью. 
 
Наивный байесовский классификатор 
Подходит  только  для  классификации.  Работает  даже  быстрее,  чем 
линейные  модели,  хорош  для  очень  больших  наборов  данных  и 
высокоразмерных данных. Часто менее точен, чем линейные модели. 
 
Деревья решений 
Очень быстрый метод, не нужно масштабировать данные, результаты 
можно визуализировать и легко объяснить. 
 
Случайные леса 
Почти  всегда  работают  лучше,  чем  одно  дерево  решений,  очень 
устойчивый  и  мощный  метод.  Не  нужно  масштабировать  данные. 
Плохо  работает  с  данными  очень  высокой  размерности  и 
разреженными данными. 
 
Градиентный бустинг деревьев решений 
Как правило, немного более точен, чем случайный лес. В отличие от 
случайного  леса  медленнее  обучается,  но  быстрее  предсказывает  и 
требует меньше памяти. По сравнению со случайным лесом требует 
настройки большего числа параметров. 
 
Метод опорных векторов 
Мощный  метод  для  работы  с  наборами  данных  среднего  размера  и 
признаками,  измеренными  в  едином  масштабе.  Требует 
масштабирования данных, чувствителен к изменению параметров. 
 











## Setup

To run the code, you need the packages ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas`` and ``pillow``.
Some of the visualizations of decision trees and neural networks structures also require ``graphviz``. The chapter
on text processing also requirs ``nltk`` and ``spacy``.

The easiest way to set up an environment is by installing [Anaconda](https://www.continuum.io/downloads).

### Installing packages with conda:
If you already have a Python environment set up, and you are using the ``conda`` package manager, you can get all packages by running

    conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz python-graphviz

For the chapter on text processing you also need to install ``nltk`` and ``spacy``:

    conda install nltk spacy


### Installing packages with pip
If you already have a Python environment and are using pip to install packages, you need to run

    pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz

You also need to install the graphiz C-library, which is easiest using a package manager.
If you are using OS X and homebrew, you can ``brew install graphviz``. If you are on Ubuntu or debian, you can ``apt-get install graphviz``.
Installing graphviz on Windows can be tricky and using conda / anaconda is recommended.
For the chapter on text processing you also need to install ``nltk`` and ``spacy``:

    pip install nltk spacy

### Downloading English language model
For the text processing chapter, you need to download the English language model for spacy using

    python -m spacy download en

