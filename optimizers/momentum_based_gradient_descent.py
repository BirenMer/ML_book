import numpy as np
from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

def do_momentum_based_gradient_descent(init_w=-2,init_b=2,eta=0.1, max_epochs=1000):
    X, Y = data_points()
    w, b = init_w,init_b
    trajectory = [(w, b)]

    prev_v_w, prev_v_b = 0, 0
    gamma = 0.9

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        # Optionally average gradients
        dw /= len(X)
        db /= len(X)

        # Momentum update
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db

        w -= v_w
        b -= v_b

        trajectory.append((w, b))

        prev_v_w = v_w
        prev_v_b = v_b

    return trajectory

def main():
    trajectory=do_momentum_based_gradient_descent()

    plot_contour(trajectory,label="Momentum GD")

if __name__ == "__main__":
    main()