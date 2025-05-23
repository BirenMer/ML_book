
import numpy as np
import matplotlib.pyplot as plt

from train_LSTM import ApplyMyLSTM, train_LSTM

#Below is the code to visualize  training data
#X_t = np.arange(-170,170,0.1)
X_t = np.arange(-70,10,0.1)
#X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1) + np.exp((X_t + 20)*0.05)

#Y_t = np.multiply(Y_t, 10*np.sin(0.1*X_t))

plt.plot(X_t, Y_t)
plt.show()


###############################################################################
#forecast Y(t) --> Y(t + dt)
from LSTM import *

dt   = 200#phase shift for prediction
[lstm, dense1, dense2] = train_LSTM(Y_t, Y_t, n_neurons = 300,\
                                   n_epoch = 1000, plot_each = 100, dt = dt,\
                                   momentum = 0.8, decay = 0.01,\
                                   learning_rate = 1e-5)


Y_hat     = ApplyMyLSTM(Y_t,lstm, dense1, dense2)
    
X_plot     = np.arange(0,len(Y_t))
X_plot_hat = np.arange(0,len(Y_hat)) + dt

plt.plot(X_plot, Y_t)
plt.plot(X_plot_hat, Y_hat)
plt.legend(['y', '$hat{y}$'])
plt.show()