import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from visualization_utils import plot_decision_regions

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

# plot data
# UNCOMMENT THE BELOW POINTS TO VIEW THE DATA GRAPH.

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')
# plt.legend(loc='upper left')

#Training the perceptron using the fit model.
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Plotting the misclassification errors versus the number of epochs
# plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')


# Mapping the different decision regions to different colors for each 
# predicted class in the grid array.

plot_decision_regions(X, y, classifier=ppn)

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')


plt.show()


# Observations : As we can see in the plot, 
# the perceptron learned a decision boundary that can classify all flower 
# examples in the Iris training subset perfectly.
