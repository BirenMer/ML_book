import math
import numpy as np
from trajectory_plotting_utils import plot_contour
from general_utils import data_points, error, grad_w,grad_b

def do_adam(init_w=-2, init_b=2, lr=0.01, max_epochs=1000, mini_batch_size=10):
    X, Y = data_points()
    w, b = init_w, init_b
    m_w, m_b = 0, 0
    v_w, v_b = 0, 0
    eps = 1e-8
    beta1, beta2 = 0.9, 0.999
    trajectory = [(w, b)]

    for epoch in range(1, max_epochs + 1):
        # Optional: shuffle data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for i in range(0, len(X), mini_batch_size):
            x_batch = X_shuffled[i:i+mini_batch_size]
            y_batch = Y_shuffled[i:i+mini_batch_size]

            dw, db = 0, 0
            for x, y in zip(x_batch, y_batch):
                dw += grad_w(w, b, x, y)
                db += grad_b(w, b, x, y)
            dw /= len(x_batch)
            db /= len(x_batch)

            # Update biased first moment estimate
            m_w = beta1 * m_w + (1 - beta1) * dw
            m_b = beta1 * m_b + (1 - beta1) * db

            # Update biased second raw moment estimate
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

            # Compute bias-corrected first moment estimate
            m_w_hat = m_w / (1 - beta1 ** epoch)
            m_b_hat = m_b / (1 - beta1 ** epoch)

            # Compute bias-corrected second moment estimate
            v_w_hat = v_w / (1 - beta2 ** epoch)
            v_b_hat = v_b / (1 - beta2 ** epoch)

            # Update parameters
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            trajectory.append((w, b))

    return trajectory
def main():
    trajectory=do_adam()
    plot_contour(trajectory=trajectory,label="Adam")

if __name__ == "__main__":
    main()