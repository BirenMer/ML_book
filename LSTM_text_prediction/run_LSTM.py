import numpy as np
import matplotlib.pyplot as plt
import random

from LSTM import LSTM 

from activation_function.Sigmoid import Sigmoid
from activation_function.Tanh import Tanh 

from optimizers.optimizerSGD import OptimizerSGD
from optimizers.optimizerSGDLSTM import OptimizerSGDLSTM

from layers.dense_layer import DenseLayer


# def RunMyLSTM(X_t, Y_t, n_epoch = 500, n_neurons = 500,\
#              learning_rate = 1e-5, decay = 0, momentum = 0.95, plot_each = 50,\
#              dt = 0):

#     #initializing LSTM
#     lstm          = LSTM(n_neurons)
#     T             = max(X_t.shape)
#     dense1        = DenseLayer(n_neurons, T)
#     dense2        = DenseLayer(T, 1)
#     optimizerLSTM = OptimizerSGDLSTM(learning_rate, decay, momentum)
#     optimizer     = OptimizerSGD(learning_rate, decay, momentum)
    
#     #Monitor   = np.zeros((n_epoch,1))
#     X_plot    = np.arange(0,T)
    
#     if dt != 0:
#         X_plots = np.arange(0,T + dt)
#         X_plots = X_plots[dt:]
#         X_t_dt  = Y_t[:-dt]
#         Y_t_dt  = Y_t[dt:]
#     else:
#         X_plots = X_plot
#         X_t_dt  = X_t
#         Y_t_dt  = Y_t
    
#     print("LSTM is running...")
    
#     for n in range(n_epoch):
        
#         if dt != 0:
#             Idx      = random.sample(range(T-dt), 2)
#             leftidx  = min(Idx)
#             rightidx = max(Idx)
            
#             X_t_cut  = X_t_dt[leftidx:rightidx]
#             Y_t_cut  = Y_t_dt[leftidx:rightidx]
#         else:
#             X_t_cut  = X_t_dt
#             Y_t_cut  = Y_t_dt
        
        
#         for i in range(5):
        
#             lstm.forward(X_t_cut)
            
#             H = np.array(lstm.H)
#             H = H.reshape((H.shape[0],H.shape[1]))
            
#             #states to Y_hat
#             dense1.forward(H[1:,:])
#             dense2.forward(dense1.output)

#             Y_hat = dense2.output
    
#             dY = Y_hat - Y_t_cut
#             #L  = 0.5*np.dot(dY.T,dY)/T_cut
            
#             dense2.backward(dY)
#             dense1.backward(dense2.dinputs)
            
#             lstm.backward(dense1.dinputs)
            
#             optimizer.pre_update_params()
#             optimizerLSTM.pre_update_params()
            
#             optimizerLSTM.update_params(lstm)
#             optimizerLSTM.post_update_params()
            
#             optimizer.update_params(dense1)
#             optimizer.update_params(dense2)
#             optimizer.post_update_params()
        
#         if not n % plot_each:
            
#             Y_hat_chunk = Y_hat

#             lstm.forward(X_t)
            
#             H = np.array(lstm.H)
#             H = H.reshape((H.shape[0],H.shape[1]))
            
#             #states to Y_hat
#             dense1.forward(H[1:,:])
#             dense2.forward(dense1.output)

#             Y_hat = dense2.output
            
#             if dt !=0:
#                 dY    = Y_hat[:-dt] - Y_t[dt:]
#             else:
#                 dY    = Y_hat - Y_t
                
#             L = 0.5*np.dot(dY.T,dY)/(T-dt)
            
#             #------------------------------------------------------------------
#             M = np.max(np.vstack((Y_hat,Y_t)))
#             m = np.min(np.vstack((Y_hat,Y_t)))
#             plt.plot(X_plot, Y_t)
#             plt.plot(X_plots, Y_hat)
#             plt.plot(X_plots[leftidx:rightidx], Y_hat_chunk)
#             plt.legend(['y', '$\hat{y}$', 'current $\hat{y}$ chunk'])
#             plt.title('epoch ' + str(n))
#             if dt != 0:
#                 plt.fill_between([X_plot[-1], X_plots[-1]],\
#                               m, M, color = 'k', alpha = 0.1)
#             plt.plot([X_plot[-1], X_plot[-1]], [m, M],'k-',linewidth = 3)
#             plt.title('epoch ' + str(n))
#             plt.show()
#             #------------------------------------------------------------------
            
#             L = float(L) 

#             print(f'current MSSE = {L:.3f}')
        
#         #updating learning rate, if decay
#         optimizerLSTM.pre_update_params()
#         optimizer.pre_update_params()
        
# ####finally, one last plot of the complete data################################
#     lstm.forward(X_t)
    
#     H = np.array(lstm.H)
#     H = H.reshape((H.shape[0],H.shape[1]))
    
#     #states to Y_hat
#     dense1.forward(H[1:,:])
#     dense2.forward(dense1.output)

#     Y_hat = dense2.output
    
#     if dt !=0:
#         dY    = Y_hat[:-dt] - Y_t[dt:]
#     else:
#         dY    = Y_hat - Y_t
                
#     L  = 0.5*np.dot(dY.T,dY)/(T-dt)
    
#     plt.plot(X_plot, Y_t)
#     plt.plot(X_plots, Y_hat)
#     plt.legend(['y', '$\hat{y}$'])
#     plt.title('epoch ' + str(n))
#     if dt != 0:
#         plt.fill_between([X_plot[-1], X_plots[-1]],\
#                       m, M, color = 'k', alpha = 0.1)
#     plt.plot([X_plot[-1], X_plot[-1]], [m, M],'k-',linewidth = 3)
#     plt.title('epoch ' + str(n))
#     plt.show()
    
    
#     L = float(L) 

#     print(f'Done! MSSE = {L:.3f}')
    
    
#     return(lstm, dense1, dense2)
def RunMyLSTM(X_t, Y_t, n_epoch=500, n_neurons=256, learning_rate=1e-3, decay=0, momentum=0.9, plot_each=50, batch_size=64):
    """
    Function to train an LSTM model for text data.

    Parameters:
    - X_t: Input sequences (shape: [n_samples, sequence_length]).
    - Y_t: Target sequences (shape: [n_samples, 1]).
    - n_epoch: Number of training epochs.
    - n_neurons: Number of LSTM neurons.
    - learning_rate: Initial learning rate.
    - decay: Learning rate decay factor.
    - momentum: Momentum for SGD optimizer.
    - plot_each: Interval for plotting progress.
    - batch_size: Size of mini-batches for training.

    Returns:
    - lstm: Trained LSTM model.
    - dense1, dense2: Trained dense layers.
    """

    # Initialize LSTM and dense layers
    sequence_length = X_t.shape[1]
    vocab_size = np.max(Y_t) + 1  # Number of unique characters

    lstm = LSTM(n_neurons)
    dense1 = DenseLayer(n_neurons, 128)
    dense2 = DenseLayer(128, vocab_size)

    optimizer_lstm = OptimizerSGDLSTM(learning_rate, decay, momentum)
    optimizer_dense = OptimizerSGD(learning_rate, decay, momentum)

    print("LSTM training started...")

    for epoch in range(n_epoch):
        # Shuffle data
        indices = np.arange(len(X_t))
        np.random.shuffle(indices)
        X_t = X_t[indices]
        Y_t = Y_t[indices]

        for i in range(0, len(X_t), batch_size):
            # Prepare batches
            X_batch = X_t[i:i + batch_size]
            Y_batch = Y_t[i:i + batch_size]

            # One-hot encode targets
            Y_one_hot = np.zeros((len(Y_batch), vocab_size))
            for j, y in enumerate(Y_batch):
                Y_one_hot[j, y] = 1

            # Forward pass
            lstm.forward(X_batch)
            H = np.array(lstm.H[1:]).reshape(len(X_batch), -1)

            dense1.forward(H)
            dense2.forward(dense1.output)

            # Compute loss (categorical cross-entropy)
            exp_scores = np.exp(dense2.output - np.max(dense2.output, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            loss = -np.mean(np.log(probs[np.arange(len(Y_batch)), Y_batch.ravel()]))

            # Backward pass
            dvalues = probs
            dvalues[np.arange(len(Y_batch)), Y_batch.ravel()] -= 1
            dvalues /= len(Y_batch)

            dense2.backward(dvalues)
            dense1.backward(dense2.dinputs)
            lstm.backward(dense1.dinputs)

            # Update parameters
            optimizer_dense.update_params(dense1)
            optimizer_dense.update_params(dense2)
            optimizer_lstm.update_params(lstm)

        # Logging progress
        if epoch % plot_each == 0 or epoch == n_epoch - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Training completed!")
    return lstm, dense1, dense2

###############################################################################
#
###############################################################################
    
def ApplyMyLSTM(X_t, lstm, dense1, dense2):
    
    T       = max(X_t.shape)
    #Y_hat   = np.zeros((T, 1))
    H       = lstm.H
    ht      = H[0]
    H       = [np.zeros((lstm.n_neurons,1)) for t in range(T+1)]
    C       = lstm.C
    ct      = C[0]
    C       = [np.zeros((lstm.n_neurons,1)) for t in range(T+1)]
    C_tilde = [np.zeros((lstm.n_neurons,1)) for t in range(T)]
    F       = [np.zeros((lstm.n_neurons,1)) for t in range(T)]
    O       = [np.zeros((lstm.n_neurons,1)) for t in range(T)]
    I       = [np.zeros((lstm.n_neurons,1)) for t in range(T)]
    
    #instances of activation functions as expected by Cell
    Sigmf    = [Sigmoid() for i in range(T)]
    Sigmi    = [Sigmoid() for i in range(T)]
    Sigmo    = [Sigmoid() for i in range(T)]
    
    Tanh1    = [Tanh() for i in range(T)]
    Tanh2    = [Tanh() for i in range(T)]
    
    #we need only the forward part
    [H, _, _, _, _, _, _, _, _, _, _] = lstm.LSTMCell(X_t, ht, ct,\
                                        Sigmf, Sigmi, Sigmo,\
                                        Tanh1, Tanh2,\
                                        H, C, F, O, I, C_tilde)
            
    
    H = np.array(H)
    H = H.reshape((H.shape[0],H.shape[1]))
    
    #states to Y_hat
    dense1.forward(H[0:-1])
    dense2.forward(dense1.output)
    
    Y_hat = dense2.output
    #plt.plot(X_t, Y_hat)
    #plt.legend(['$\hat{y}$'])
    #plt.show()
    
    return(Y_hat)