import numpy as np

"""
We will try to build a text generation model using an RNN.
We train our model to predict the probability of a character given the preceding characters. 
It's a generative model.
"""


class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        # Defining the hyper params:
        self.hidden_size = hidden_size
        # Number of unique tokens in your dataset.
        self.vocab_size = vocab_size
        #: Length of input sequences processed by the RNN.
        self.seq_length = seq_length

        self.learning_rate = learning_rate

        """
        Recommended approach is to initialize the weights randomly in the interval from
        [ -1/sqrt(n),1/sqrt(n)]
        where n is the number of incoming connections from the previous layer.
        """
        # Defining model parameters
        #  U: Weight matrix for input to hidden layer
        self.U = np.random.uniform(
            -np.sqrt(1.0 / vocab_size),
            np.sqrt(1.0 / vocab_size),
            (hidden_size, vocab_size),
        )
        # V :Weight matrix for hidden to output layer.
        self.V = np.random.uniform(
            -np.sqrt(1.0 / hidden_size),
            np.sqrt(1.0 / hidden_size),
            (vocab_size, hidden_size),
        )
        # W: Weight matrix for hidden state transition
        self.W = np.random.uniform(
            -np.sqrt(1.0 / hidden_size),
            np.sqrt(1.0 / hidden_size),
            (hidden_size, hidden_size),
        )
        self.b = np.zeros((hidden_size, 1))  # Bias for hidden layer
        self.c = np.zeros((vocab_size, 1))  # Bias for output layer

    # Defnining the forward pass
    def forward(self, inputs, hprev):
        """
        Forward pass through the RNN.

        t :timestamp
        hs[t] : hidden state for timestamp t
        os[t] : output for timestamp t


        inputs : list of integers (indices) representing the input sequence
        hprev  : previous hidden state
        returns:
            xs   : input vectors (one-hot encoded)
            hs    : hidden states

        """
        xs, hs, os, ycap = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = np.zeros(self.vocab_size, 1)
            xs[t][inputs[t]] = 1  # one hot encoding,1-of-k
            
            # Compute the hidden state
            hs[t] = np.tanh(
                np.dot(self.U, xs[t]) + np.dot(self.W, hs[t - 1]) + self.b
            )  
            
            # Compute the output logits
            os[t] = (
                np.dot(self.V, hs[t]) + self.c
            )  
            
            # Apply softmax to get the probabilities for the next character
            ycap[t] = self.softmax(
                os[t]
            )  
        return xs, hs, ycap

    # implementing the softmax function
    def softmax(self, x):
        p = np.exp(x - np.max(x))
        return p / np.sum(p)

    # defining the loss function
    def loss(self, ps, targets):
        """loss of sequence"""
        # Calculating cross-entropy loss
        return sum(-np.log(ps[t][targets[t], 0] for t in range(self.seq_length)))

    # Defining the backward pass / back propogation BPTT
    def backward(self, xs, hs, ycap, targets):
        # backward pass computer gradients
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(self.seq_length)):
            # Start with output
            dy = np.copy(ycap[t])

            # gradient through softmax
            dy[targets[t]] -= 1

            # Updating dV and dc
            dV += np.dot(dy, hs[t].T)
            dc += dc

            # dh has two components gradient flowing from output and from next cell
            dh = np.dot(self.V.T, dy) + dh_next  # backprop into h

            # dh_rec is the recurring component seen in most of the calculations
            dh_rec = (1 - hs[t] * hs[t]) * dh  # backprop through tanh non-linearity
            dh += dh_rec

            # Updating dU and dW
            dU += np.dot(dh_rec, xs[t].T)
            dW += np.dot(dh_rec, hs[t - 1].T)

            # pass the gradient fromt the next cell for next iteration
            dh_next = np.dot(self.W.T, dh_rec)

        # Clipping weights to avoid gradient explosion.
        for d_param in [dU, dW, dV, db, dc]:
            np.clip(d_param, -5, 5, out=d_param)
        return dU, dW, dV, db, dc

    """
    Using BPTT we calculated the gradient for each parameter of the model. it is now time to update the weights.
    """

    # Defining the weight update function
    def update_model(self, dU, dV, dW, db, dc):
        for param, d_param in zip(
            [self.U, self.W, self.V, self.b, self.c], [dU, dV, dW, db, dc]
        ):
            # Changing paramerters according to gradients and learning rate
            param += -self.learning_rate * d_param
