import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pd.options.display.max_columns = 10

df = pd.read_csv('titanic.csv')
#print(df.head()) displays 5 data awal
#print(df.describe()) show mean std etc
#yo = df[['Umur','Uang']] call column
#df['male'] = df['Sex'] == 'male' #ubah kolom sex yang male jadi true
arr = df[['Pclass','Fare','Age']].values #[:10] taking the first 10 values
mask = arr[:,2] < 18
#print(mask.sum())
plt.xlabel('Age')
plt.ylabel('Fare')
plt.scatter(df['Age'],df['Fare'],c = df['Survived'])
plt.show()
plt.colorbar()
