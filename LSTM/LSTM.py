import numpy as np
from Tanh import Tanh
from Sigmoid import Sigmoid


class LSTM:
    def __init__(self, n_neurons) -> None:
        # input is number of neurons / number of stats
        self.n_neurons = n_neurons

        # Defining forget gate
        self.Uf = 0.1 * np.random.randn(
            n_neurons, 1
        )  # Size of Uf We can change 1 if we want to have more the on feature lstm
        self.bf = 0.1 * np.random.randn(n_neurons, 1)  # Bias for Forget Gate
        self.Wf = 0.1 * np.random.randn(n_neurons, n_neurons)  # Weight for Forget Gate

        # Defining input gate
        self.Ui = 0.1 * np.random.randn(n_neurons, 1)
        self.bi = 0.1 * np.random.randn(n_neurons, 1)
        self.Wi = 0.1 * np.random.randn(n_neurons, n_neurons)

        # Defining output gate
        self.Uo = 0.1 * np.random.randn(n_neurons, 1)
        self.bo = 0.1 * np.random.randn(n_neurons, 1)
        self.Wo = 0.1 * np.random.randn(n_neurons, n_neurons)

        # Defining the c tilde (or c dash)
        self.Ug = 0.1 * np.random.randn(n_neurons, 1)
        self.bg = 0.1 * np.random.randn(n_neurons, 1)
        self.Wg = 0.1 * np.random.randn(n_neurons, n_neurons)

    # defining the forward pass function
    def forward(self, X_t):
        T = max(X_t.shape)

        self.T = T
        self.X_t = X_t

        n_neurons = self.n_neurons

        # We are doing this as we would like to keep track of H,C and C_tilde as well as forget gate, input gate and output gate
        self.H = [
            np.zeros((n_neurons, 1)) for t in range(T + 1)
        ]  # Adding values from the first timestamp to the last time stamp
        self.C = [np.zeros((n_neurons, 1)) for t in range(T + 1)]
        self.C_tilde = [
            np.zeros((n_neurons, 1)) for t in range(T)
        ]  # last -1 time stamp

        # This part is helpful for debugging we really don't need this in code
        self.F = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.I = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.O = [np.zeros((n_neurons, 1)) for t in range(T)]

        # Now for the gates we would like to change the values of the learnable with our optimizers so we define them with d as prefix
        # Forget Gate
        self.dUf = 0.1 * np.random.randn(n_neurons, 1)
        self.dbf = 0.1 * np.random.randn(n_neurons, 1)
        self.dWf = 0.1 * np.random.randn(n_neurons, n_neurons)

        # input Gate
        self.dUi = 0.1 * np.random.randn(n_neurons, 1)
        self.dbi = 0.1 * np.random.randn(n_neurons, 1)
        self.dWi = 0.1 * np.random.randn(n_neurons, n_neurons)

        # output Gate
        self.dUo = 0.1 * np.random.randn(n_neurons, 1)
        self.dbo = 0.1 * np.random.randn(n_neurons, 1)
        self.dWo = 0.1 * np.random.randn(n_neurons, n_neurons)

        # c_tilde
        self.dUg = 0.1 * np.random.randn(n_neurons, 1)
        self.dbg = 0.1 * np.random.randn(n_neurons, 1)
        self.dWg = 0.1 * np.random.randn(n_neurons, n_neurons)

        # For every timestamp we create an output and then we want to run back propogation through time

        # Now we initializing all the matrices for the backprop function
        # We still need to define activations function like sigmoid and tanh
        Sigmf = [Sigmoid() for i in range(T)]
        Sigmi = [Sigmoid() for i in range(T)]
        Sigmo = [Sigmoid() for i in range(T)]

        Tanh1 = [Tanh() for i in range(T)]
        Tanh2 = [Tanh() for i in range(T)]

        ht = self.H[0]  # 0th time stamp
        ct = self.C[0]  # 0th time stamp

        # Creating the LSTM CELL
        [H, C, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2, F, I, O, C_tilde] = self.LSTMCell(
            X_t,
            ht,
            ct,
            Sigmf,
            Sigmi,
            Sigmo,
            Tanh1,
            Tanh2,
            self.H,
            self.C,
            self.F,
            self.O,
            self.I,
            self.C_tilde,
        )

        self.F = F
        self.O = O
        self.I = I
        self.C_tilde = C_tilde

        self.H = H
        self.C = C

        self.Sigmf = Sigmf
        self.Sigmi = Sigmi
        self.Sigmo = Sigmo

        self.Tanh1 = Tanh1
        self.Tanh2 = Tanh2

    def LSTMCell(
        self, X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2, H, C, F, O, I, C_tilde
    ):
        for t,xt in enumerate(X_t):
            xt=xt.reshape(1,1)
            # Coding the equation for forget gate
            outf=np.dot(self.Uf,xt)+np.dot(self.Wf,ht)+self.bf
            Sigmf[t].forward(outf)
            ft=Sigmf[t].output

            #Coding the equation for input gate
            outi=np.dot(self.Ui,xt)+np.dot(self.Wi,ht)+self.bi
            Sigmi[t].forward(outi)
            it=Sigmi[t].output

            #Coding the equation for output gate
            outo=np.dot(self.Uo,xt)+np.dot(self.Wo,ht)+self.bo
            Sigmo[t].forward(outo)
            ot=Sigmo[t].output

            #Coding the equation for C_tilde
            outct_tilde=np.dot(self.Ug,xt)+np.dot(self.Wg,ht)+self.bg
            Tanh1[t].forward(outct_tilde)
            ct_tilde=Tanh1[t].output

            #Combining the infromation from the input gat and forget gate with c_tilde
            #using multiply as it is an element wise operation
            ct=np.multiply(ft,ct)+np.multiply(it,ct_tilde)

            #passing it to our second tanh activation function
            Tanh2[t].forward(ct)
            ht=np.multiply(Tanh2[t].output,ot)

            #storing the outputs
            H[t+1]=ht
            C[t+1]=ct
            C_tilde[t]=ct_tilde

            F[t]=ft
            I[t]=it
            O[t]=ot

        return (H,C,Sigmf,Sigmi,Sigmo,Tanh1,Tanh2,F,I,O,C_tilde)