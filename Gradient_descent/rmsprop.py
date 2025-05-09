
import numpy as np
from trajectory_plotting_utils import plot_contour
from general_utils import data_points, error, grad_w,grad_b

def do_rmsprop(init_w=-2, init_b=2, lr=0.1, max_epochs=1000):
    w, b = init_w, init_b
    eta = lr
    v_w, v_b = 0, 0
    eps = 1e-8
    beta = 0.9
    X, Y = data_points()
    trajectory = [(w, b)]

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        dw /= len(X)
        db /= len(X)

        v_w = beta * v_w + (1 - beta) * (dw ** 2)
        v_b = beta * v_b + (1 - beta) * (db ** 2)

        w = w - (eta / np.sqrt(v_w + eps)) * dw
        b = b - (eta / np.sqrt(v_b + eps)) * db

        trajectory.append((w, b))

    return trajectory
trajectory=do_rmsprop()
plot_contour(trajectory=trajectory,label="RMSPROP Path")