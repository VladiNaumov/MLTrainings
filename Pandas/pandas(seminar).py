#!/usr/bin/env python
# coding: utf-8

# Библиотека `pandas` активно используется в современном data science для работы с данными, которые могут быть представлены в виде таблиц (а это очень, очень большая часть данных)

# `pandas` есть в пакете Anaconda, но если вдруг у Вас её по каким-то причинам нет, то можно установить, раскомментировав одну из следующих команд:

# In[ ]:


# !pip3 install pandas
# !conda install pandas


# In[ ]:


import numpy as np
import pandas as pd # Стандартное сокращение для pandas. Всегда используйте его!


# # pd.Series
# 
# Тип данных pd.Series представляет собой одномерный набор данных. Отсутствующий данные записываются как `np.nan` (в этот день термометр сломался или метеоролог был пьян); они не участвуют в вычислении средних, среднеквадратичных отклонений и т.д.
# 
# ### Создание
# Создадим Series из списка температур

# In[ ]:


some_list = [1, 3, 5, np.nan, 6, 8]
ser_1 = pd.Series(some_list)
ser_1


# In[ ]:


# Так же можно в явном виде указать индексы, чтобы потом было более удобно обращаться к элементам
ind = ['1st day', '2nd day', '3rd day', '4th day', '5rd day', '6th day']

ser_2 = pd.Series(some_list, index=ind)
ser_2


# In[ ]:


ser_2['4th day']


# In[ ]:


# А еще можно дать pd.Series имя, чтобы было совсем красиво
ser_3 = pd.Series(some_list, index=ind, name='Temperature')
ser_3


# ### Индексирование
# С индексами можно работать так же, как и в случае с обычным list.

# In[ ]:


print(ser_3[0])

print('-----------')

print(ser_3[1:3])

print('-----------')

print(ser_3[::-1])


# ### Индексирование pd.Series по условиям

# In[ ]:


date_range = pd.date_range('20190101', periods=10)
ser_4 = pd.Series(np.random.rand(10), index=date_range)
ser_4


# In[ ]:


ser_4 > 0.5


# В качестве индекса можно указать выражение, и нам будут возвращены только те элементы, для которых значение является `True`

# In[ ]:


ser_4[ser_4 > 0.5]


# In[ ]:


ser_4[(ser_4 > 0.6) | (ser_4 < 0.2)]


# In[ ]:


ser_4[(ser_4 > 0.6) & (ser_4 < 0.2)]


# ### Сортировки
# Тип `pd.Series` можно отсортировать как по значениям, так и по индексу.

# In[ ]:


ser_4.sort_index()


# In[ ]:


ser_4 = ser_4.sort_values()


# In[ ]:


ser_4


# ### Операции с series
# Тип `pd.Series` можно модифицировать проще, чем стандартный ``list`` из Python.

# In[ ]:


ser_4 + 100


# In[ ]:


np.exp(ser_4)


# In[ ]:


term_1 = pd.Series(np.random.randint(0, 10, 5))
term_2 = pd.Series(np.random.randint(0, 10, 6))

term_1 + term_2


# In[ ]:


term_1.shape


# # pd.DataFrame
# 
# Тип данных pd.DataFrame представляет собой двумерную таблицу с данными. Имеет индекс и набор столбцов (возможно, имеющих разные типы). Таблицу можно построить, например, из словаря, значениями в котором являются одномерные наборы данных.
# ### Создание и основные объекты

# In[ ]:


# Dataframe можно составить из словаря. Ключ будет соответсовать колонке
some_dict = {'one': pd.Series([1,2,3], index=['a','b','c']),
             'two': pd.Series([1,2,3,4], index=['a','b','c','d']),
             'three': pd.Series([5,6,7,8], index=['a','b','c','d'])}
df = pd.DataFrame(some_dict)
df


# In[ ]:


#Альтернативно, из списка списков с аргументом columns

some_array = [[1,1,5], [2,2,6], [3,3,7], [np.nan, 4,8]]
df = pd.DataFrame(some_array, index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
df


# In[ ]:


df.values


# In[ ]:


df.columns


# In[ ]:


df.columns = ['first_column', 'second_column', 'third_column']
df.index = [1,2,3,4]
df


# ### Индексирование 
# Есть очень много способов индексировать DataFrame в Pandas. Не все из них хорошие! Вот несколько удобных, но не универсальных.
# 
# #### По колонкам
# Индексирование по колонке возращает pd.Series. Можно выбирать не одну колонку, а сразу несколько. Тогда снова вернётся pd.DataFrame.

# In[ ]:


first_column = df['first_column']
first_column


# In[ ]:


df.first_column


# In[ ]:


subset_dataframe = df[['first_column', 'second_column']]
subset_dataframe


# In[ ]:


one_column_dataframe = df[['first_column']]
one_column_dataframe


# #### По строкам
# Можно писать любые слайсы, как в Python-списке. Они будут применяться к строкам. Нельзя обращаться по элементу!

# In[ ]:


df[1] # не сработает


# In[ ]:


df[:1]


# In[ ]:


df[1:4]


# #### Универсальное индексирование: .loc и .iloc
# 
# .loc и .iloc --- это два взаимозаменяемых атрибута, которые позволяют индексировать по обеим осям сразу. Путаницы не возникает из-за фиксированного порядка перечисления осей.

# In[ ]:


# По индексам: 
df.iloc[1:3, :2]


# In[ ]:


df.loc[1:3, ['first_column', 'second_column']]


# Лучше использовать по умолчанию либо только loc, либо только .iloc! А лучше вообще всегда только .iloc, чтобы не запутаться.

# ### Модификации датасета, создание новых колонок
# Можно просто брать и создавать новую колонку. Синтаксис тут вполне естественный.

# In[ ]:


new_column = [5,2,1,4]
df['new_column'] = new_column
df


# Аналогично, можно применять к отдельным колонкам арифметические операции (ведь колонки --- это Series!)

# In[ ]:


df['first_column'] = df['first_column'] * 10
df


# In[ ]:


Информация о файлах:

titanic_data.csv содержит различную информацию о пассажирах Титаника (билет, класс, возраст и т.п.)
titanic_surv.csv содержит для каждого пассажира из первого файла информацию о том, выжил ли этот пассажир (метка 1) или нет (метка 0)
Чтение из файла
Обычно данные хранятся в виде таблиц в файлах формата .csv или .xlsx. На этом семинаре мы будем загружать данные из .csv файлов.

Загрузим первый файл


# In[ ]:


# df_1 = pd.read_csv('titanic_data.csv')
pass_link = 'https://www.dropbox.com/s/lyzcuxu1pdrw5qb/titanic_data.csv?dl=1'
titanic_passengers = pd.read_csv(pass_link, index_col='PassengerId') # index_col=?


# In[ ]:


print('Всего пассажиров: ', len(titanic_passengers))
titanic_passengers.head(10)


# ### Разная информация о датасете
# 
# Можно узнать размер таблицы, информацию о значениях таблицы, различные статистики по значениям.

# In[ ]:


titanic_passengers.shape


# In[ ]:


titanic_passengers.info()


# In[ ]:


titanic_passengers.describe()


# ## Задание 1 
# Опишите данный датасет: какое расределение женщин/мужчин в нем? Сколько пассажиров ехало в каждом классе? Какой средний/минимальный/максимальный возраст пассажиров?

# In[ ]:


(titanic_passengers['Age'].min(), titanic_passengers['Age'].mean(), titanic_passengers['Age'].max())


# In[ ]:


titanic_passengers['Sex'].value_counts()


# In[ ]:


titanic_passengers['Pclass'].value_counts()


# ## Задание 2
# Сгруппируйте записи по классам пассажиров, в каждой группе посчитайте средний возраст. Используйте метод ``pandas.DataFrame.groupby``.

# In[ ]:


titanic_passengers.groupby(['Pclass']).mean()


# In[ ]:


titanic_passengers.groupby(['Pclass'])['Age'].mean()


# ## Слияние таблиц
# Таблицы можно сливать несколькими способами. Мы рассмотрим слияние по индексу: метод называется ``pd.join``.

# In[ ]:


# df_2 = pd.read_csv('titanic_surv.csv')
surv_link = 'https://www.dropbox.com/s/v35x9i6a1tc7emm/titanic_surv.csv?dl=1'
df_2 = pd.read_csv(surv_link)


# In[ ]:


df_2.head()


# ### Задание 3.
# Слейте два датасета по колонке индекса.

# In[ ]:


df_2.index = np.arange(1, 892)


# In[ ]:


df_2 = df_2.sample(frac=1)
df_2.head()


# In[ ]:


titanic_passengers = titanic_passengers.join(df_2)
titanic_passengers.head()


# ### Задание 4. 
# Сколько всего выживших пассажиров? Выживших пассажиров по каждому из полов? Постройте матрицу корреляций факта выживания, пола и возраста.

# In[ ]:


titanic_passengers['Survived'].sum()


# In[ ]:


titanic_passengers.groupby(['Sex'])['Survived'].sum()


# In[ ]:


corr_data = titanic_passengers[['Sex', 'Age', 'Survived']]
corr_data['Sex'] = (corr_data['Sex'] == 'female').astype(int)


# In[ ]:


corr_data.head()


# In[ ]:


corr_data.corr()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, annot_kws={"size": 16})

