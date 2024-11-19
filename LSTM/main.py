import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM
from dense_layer import DenseLayer

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
lstm.forward(X_t)

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
    #FIX  Explanation:
    # np.dot(dY.T, dY).item() extracts the scalar value from the (1, 1) array resulting from the dot product.
    # This ensures the computation proceeds as intended without warnings.
    L = float(0.5 * np.dot(dY.T, dY).item() / T)

    Monitor[n]=L


    dense2.backward(dY)
    dense1.backward(dense2.dinputs)
    lstm.backward(dense1.dinputs)

    dense1.weights-=lr*dense1.dweights
    dense2.weights-=lr*dense2.dweights

    dense1.biases-=lr*dense1.dbiases
    dense2.biases-=lr*dense2.dbiases

    lstm.Uf-=lr*lstm.dUf
    lstm.Ui-=lr*lstm.dUi
    lstm.Uo-=lr*lstm.dUo
    lstm.Ug-=lr*lstm.dUg

    lstm.Wf-=lr*lstm.dWf
    lstm.Wi-=lr*lstm.dWi
    lstm.Wo-=lr*lstm.dWo
    lstm.Wg-=lr*lstm.dWg

    lstm.bf-=lr*lstm.dbf
    lstm.bi-=lr*lstm.dbi
    lstm.bo-=lr*lstm.dbo
    lstm.bg-=lr*lstm.dbg
    print(f"Epoch {n} and Current MSE Loss = {L:0.3f}")
plt.plot(range(n_epoch),Monitor)
plt.xlabel('epochs')
plt.ylabel('MSSE')
plt.yscale('log')
plt.show()
print("DONE")



