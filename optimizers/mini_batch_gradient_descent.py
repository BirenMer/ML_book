import numpy as np
from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

def do_mini_batch_gradient_descent(init_w=-2, init_b=2, lr=0.1, max_epochs=1000, mini_batch_size=2):
    w, b = init_w,init_b
    eta = lr
    X, Y = data_points()
    trajectory = [(w, b)]
    
    for epoch in range(max_epochs):
        dw, db, num_points_seen = 0, 0, 0
        
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
            num_points_seen += 1

            if num_points_seen % mini_batch_size == 0:

                # Update weights after each mini-batch
                w -= eta * dw
                b -= eta * db
                trajectory.append((w, b))
                dw, db = 0, 0  # Reset gradients

        # if dataset size isn't divisible by mini_batch_size
        if num_points_seen % mini_batch_size != 0:
            w -= eta * dw
            b -= eta * db
            trajectory.append((w, b))

    return trajectory

def main():
    trajectory=do_mini_batch_gradient_descent()
    plot_contour(trajectory=trajectory,label="Mini Batch GD")

if __name__ == "__main__":
    main()