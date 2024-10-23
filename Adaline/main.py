import numpy as np
import matplotlib.pyplot as plt 
from adaline import AdalineGD
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

s = 'https://archive.ics.uci.edu/ml/'\
     'machine-learning-databases/iris/iris.data'
print('From URL:', s)
# From URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.
# data
df = pd.read_csv(s, header=None, encoding='utf-8')
# print(df)
print(df.tail())


# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# Creating the first adaline instance with lr = 0.1 
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)

ax[0].plot(range(1, len(ada1.losses_) + 1),
np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')
# Creating the second adaline instance with lr=0.0001

ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1),
ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()