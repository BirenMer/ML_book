import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM
from dense_layer import DenseLayer
from Optimizer_SGD import OptimizerSGD, OptimizerSGDLSTM
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

# plt.plot(X_t,Y_t)

# plt.show()
#Hyper Params
n_neurons=200
lr=1e-5
n_epoch=100


lstm=LSTM(n_neurons)

#Creating a plot to see the code working CKP-1
# for h in lstm.H:
#     plt.plot(np.arange(20),h[0:20],'k-',linewidth=1,alpha=0.05)


# for c in lstm.C:
#     plt.plot(np.arange(20),h[0:20],'k-',linewidth=1,alpha=0.05)
# plt.show()

#Adding full back prop CKP-2
T=max(X_t.shape)
# no of features in our case is the number of neurons in the first layer
dense1=DenseLayer(n_neurons,T)

#Note the number of neurons in the first layer has to be same as the number of inputs in the second layer 

# Setting up second dense layer which can convert the inpput into prediction Y hat
dense2=DenseLayer(T,1)

#setting up optimizer for lstm
optimizer_lstm=OptimizerSGDLSTM()

#setting up optimizer for dense layer
optimizer=OptimizerSGD()

Monitor=np.zeros((100))

for n in range(n_epoch):
    #Calling the forward part for n_epoch
    lstm.forward(X_t)
    H=np.array(lstm.H)
    H=H.reshape((H.shape[0],H.shape[1]))

    dense1.forward(H[1:,:])
    dense2.forward(dense1.output)
    
    Y_hat=dense2.output
    dY=Y_hat-Y_t
    
    # L=float(0.5*np.dot(dY.T,dY)/T) #Depricated 
    
    # FIX  Explanation:
    # np.dot(dY.T, dY).item() extracts the scalar value from the (1, 1) array resulting from the dot product.

    # This ensures the computation proceeds as intended without warnings.
    L = float(0.5 * np.dot(dY.T, dY).item() / T)

    Monitor[n]=L


    dense2.backward(dY)
    dense1.backward(dense2.dinputs)
    lstm.backward(dense1.dinputs)
    #adding optimizer
    optimizer_lstm.pre_update_params()
    optimizer.pre_update_params()

    #this will look for layer.dwights and layer.dbiases and will update the same
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    #donign the same for lstm
    optimizer_lstm.update_params(lstm)

    optimizer_lstm.post_update_params()
    optimizer.post_update_params()

    print(f"Epoch {n} and Current MSE Loss = {L:0.3f}")
plt.plot(range(n_epoch),Monitor)
plt.xlabel('epochs')
plt.ylabel('MSSE')
plt.yscale('log')
plt.show()
print("DONE")



