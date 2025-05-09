import numpy as np

from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

def do_stochastic_momentum_based_gradient_descent(init_w=-2, init_b=2, lr=0.1, max_epochs=1000):
    w, b = init_w, init_b
    eta = lr
    gamma = 0.9
    prev_v_w, prev_v_b = 0, 0
    X, Y = data_points()
    trajectory = [(w, b)]

    for epoch in range(max_epochs):
        for x, y in zip(X, Y):
            dw = grad_w(w, b, x, y)
            db = grad_b(w, b, x, y)

            v_w = gamma * prev_v_w + eta * dw
            v_b = gamma * prev_v_b + eta * db

            w = w - v_w
            b = b - v_b

            prev_v_w = v_w
            prev_v_b = v_b

            trajectory.append((w, b))

    return trajectory
trajectory=do_stochastic_momentum_based_gradient_descent()
plot_contour(trajectory=trajectory,label="Stochastic Momentum Based")