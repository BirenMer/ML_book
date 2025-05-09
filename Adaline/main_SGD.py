import numpy as np
import matplotlib.pyplot as plt 
from adalineSGD import AdalineSGD

import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from utils.visualization_utils import plot_decision_regions

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

# Adding Standardization using the built-in NumPy methods mean and std.
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


ada_sgd = AdalineSGD(n_iter=20, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()