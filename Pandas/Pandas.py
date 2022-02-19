import pandas as pd

"""
columns = ['Contry', 'Provision', 'Region']
index = [1, 2, 3]
data = [['italy', 'France', 'Finland'],
        ['litva', 'Numr', 'GFG'],
        ['REW', 'SEK', 'DOG']]

df = pd.DataFrame (data, columns, index )
print(df)

"""

df_cv = pd.read_csv('C:/Users/Sim/Desktop/MLTrainings/data/ram_price.csv')
print(df_cv)

print(df_cv.count())





