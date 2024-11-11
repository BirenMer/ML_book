import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM

#Creating data 

X_t=np.arange(-70,10,0.1)
# X_t=np.arange(-10,10,0.1)
X_t=X_t.reshape(len(X_t),1)

"""
Y_t represents a sinusoidal wave with noise.
* Sine Wave (np.sin(X_t)
* Random Noise (0.1 * np.random.randn(len(X_t), 1))
* Exponential Growth (np.exp((0.5 * X_t + 20) * 0.05))
"""
Y_t = np.sin(X_t)+0.1*np.random.randn(len(X_t),1)+np.exp((0.5*X_t+20)*0.05)

plt.plot(X_t,Y_t)

plt.show()

lstm=LSTM(n_neurons=200)
lstm.forward(X_t)
