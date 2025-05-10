#WIP code
import numpy as np

from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

def do_nesterov_accelerated_gradient_descent(init_w=-2,init_b=2,lr=0.1, max_epochs=1000):
    X, Y = data_points()
    w, b = init_w,init_b
    eta = lr
    gamma = 0.9

    prev_v_w, prev_v_b = 0, 0
    trajectory = [(w, b)]

    for i in range(max_epochs):
        # Look ahead
        lookahead_w = w - gamma * prev_v_w
        lookahead_b = b - gamma * prev_v_b

        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(lookahead_w, lookahead_b, x, y)
            db += grad_b(lookahead_w, lookahead_b, x, y)

        # Optional: average gradients
        dw /= len(X)
        db /= len(X)

        # Update velocities
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db

        # Update parameters
        w -= v_w
        b -= v_b

        # Save and update previous velocities
        trajectory.append((w, b))
        prev_v_w = v_w
        prev_v_b = v_b

    return trajectory

def main():
    trajectory=do_nesterov_accelerated_gradient_descent()  
    plot_contour(trajectory,label="Nesterov Accelerated GB")

if __name__ == "__main__":
    main()