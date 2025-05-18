import numpy as np
import matplotlib.pyplot as plt

class SimpleGRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W_z = np.random.randn(hidden_size, input_size)
        self.U_z = np.random.randn(hidden_size, hidden_size)
        self.b_z = np.zeros((hidden_size, 1))
        
        self.W_r = np.random.randn(hidden_size, input_size)
        self.U_r = np.random.randn(hidden_size, hidden_size)
        self.b_r = np.zeros((hidden_size, 1))
        
        self.W_h = np.random.randn(hidden_size, input_size)
        self.U_h = np.random.randn(hidden_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))
        
        self.W_y = np.random.randn(output_size, hidden_size)
        self.b_y = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def forward(self, x):
        T = len(x)
        h_prev = np.zeros((self.hidden_size, 1))  # Initial hidden state
        h_list = []  # Stores previous h for each time step
        z_list = []
        r_list = []
        h_tilde_list = []
        y_list = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)

            # Update gate
            z = self.sigmoid(np.dot(self.W_z, x_t) + np.dot(self.U_z, h_prev) + self.b_z)
            z_list.append(z)

            # Reset gate
            r = self.sigmoid(np.dot(self.W_r, x_t) + np.dot(self.U_r, h_prev) + self.b_r)
            r_list.append(r)

            # Candidate hidden state
            h_tilde = self.tanh(np.dot(self.W_h, x_t) + np.dot(self.U_h, r * h_prev) + self.b_h)
            h_tilde_list.append(h_tilde)

            # Hidden state update
            h = (1 - z) * h_prev + z * h_tilde

            # Output
            y = np.dot(self.W_y, h) + self.b_y
            y_list.append(y)

            # Store previous h for this time step
            h_list.append(h_prev)
            h_prev = h  # Update h_prev for next time step

        return y_list, h_list, z_list, r_list, h_tilde_list

    def backward(self, x, y_list, target, h_list, z_list, r_list, h_tilde_list):
        T = len(x)
        dW_z = np.zeros_like(self.W_z)
        dU_z = np.zeros_like(self.U_z)
        db_z = np.zeros_like(self.b_z)
        
        dW_r = np.zeros_like(self.W_r)
        dU_r = np.zeros_like(self.U_r)
        db_r = np.zeros_like(self.b_r)
        
        dW_h = np.zeros_like(self.W_h)
        dU_h = np.zeros_like(self.U_h)
        db_h = np.zeros_like(self.b_h)
        
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x_t = x[t].reshape(-1, 1)
            target_t = target[t].reshape(-1, 1)
            y = y_list[t]
            h_prev = h_list[t]
            z = z_list[t]
            r = r_list[t]
            h_tilde = h_tilde_list[t]

            # Output gradient
            dy = y - target_t
            dW_y += np.dot(dy, h_prev.T)
            db_y += dy

            # Gradient from output to hidden state
            dh = np.dot(self.W_y.T, dy) + dh_next

            # Gradients for h_tilde and h_prev
            dh_prev = dh * (1 - z)
            dh_tilde = dh * z

            # Gradient for h_tilde
            dh_tanh = dh_tilde * (1 - h_tilde ** 2)
            dW_h += np.dot(dh_tanh, x_t.T)
            dU_h += np.dot(dh_tanh, (r * h_prev).T)
            db_h += dh_tanh

            # Gradient for r
            dr = np.dot(self.U_h.T, dh_tanh) * h_prev
            dr_sigmoid = dr * r * (1 - r)
            dW_r += np.dot(dr_sigmoid, x_t.T)
            dU_r += np.dot(dr_sigmoid, h_prev.T)
            db_r += dr_sigmoid

            # Gradient for z
            dz = dh * (h_tilde - h_prev)
            dz_sigmoid = dz * z * (1 - z)
            dW_z += np.dot(dz_sigmoid, x_t.T)
            dU_z += np.dot(dz_sigmoid, h_prev.T)
            db_z += dz_sigmoid

            # Update dh_next for previous time step
            dh_next = dh_prev + np.dot(self.U_z.T, dz_sigmoid) + np.dot(self.U_r.T, dr_sigmoid)

        return dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_y, db_y

    def update_parameters(self, dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_y, db_y, learning_rate):
        self.W_z -= learning_rate * dW_z
        self.U_z -= learning_rate * dU_z
        self.b_z -= learning_rate * db_z
        
        self.W_r -= learning_rate * dW_r
        self.U_r -= learning_rate * dU_r
        self.b_r -= learning_rate * db_r
        
        self.W_h -= learning_rate * dW_h
        self.U_h -= learning_rate * dU_h
        self.b_h -= learning_rate * db_h
        
        self.W_y -= learning_rate * dW_y
        self.b_y -= learning_rate * db_y

    def predict(self, x):
        y_list, _, _, _, _ = self.forward(x)
        predictions = [self.softmax(y) for y in y_list]
        return predictions

    def visualize(self, loss_history, predictions, targets):
        plt.figure(figsize=(12, 6))

        # Plot loss over epochs
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Plot predicted vs actual outputs for the last time step
        plt.subplot(1, 2, 2)
        last_prediction = predictions[-1].flatten()
        last_target = targets[-1].flatten()
        
        x_labels = [f'Class {i+1}' for i in range(len(last_target))]
        x = np.arange(len(last_target))
        width = 0.35
        
        plt.bar(x - width/2, last_target, width, label='Actual')
        plt.bar(x + width/2, last_prediction, width, label='Predicted')
        
        plt.title('Predicted vs Actual Outputs (Last Time Step)')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.xticks(x, x_labels)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
input_size = 4
hidden_size = 3
output_size = 2

gru = SimpleGRU(input_size, hidden_size, output_size)

# Generate random data
sequence_length = 5
data = [np.random.randn(input_size) for _ in range(sequence_length)]
target = [np.random.randn(output_size) for _ in range(sequence_length)]

# Training loop example
loss_history = []
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_list, h_list, z_list, r_list, h_tilde_list = gru.forward(data)
    
    # Calculate loss (mean squared error)
    loss = 0
    for t in range(sequence_length):
        loss += np.mean((y_list[t] - target[t].reshape(-1, 1))**2)
    loss /= sequence_length
    loss_history.append(loss)
    
    # Backward pass
    dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_y, db_y = gru.backward(data, y_list, target, h_list, z_list, r_list, h_tilde_list)
    
    # Update weights and biases
    gru.update_parameters(dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_y, db_y, learning_rate)

# Make predictions
predictions = gru.predict(data)

# Visualize results
gru.visualize(loss_history, predictions, target)