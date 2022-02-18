import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

s = 'C:\\Users\\Sim\\Desktop\\DataScience\\app\\Book\\iris.data'
# print(s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

df.tail()

df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()

