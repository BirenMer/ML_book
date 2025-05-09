import numpy as np

from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

def do_adagrad(init_w=-2, init_b=2, max_epochs=1000, lr=0.1):
    w, b, eta = init_w, init_b, lr
    v_w, v_b, eps = 0, 0, 1e-8
    X, Y = data_points()
    trajectory = [(w, b)]

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        dw /= len(X)
        db /= len(X)

        v_w += dw ** 2
        v_b += db ** 2

        w = w - (eta / np.sqrt(v_w + eps)) * dw
        b = b - (eta / np.sqrt(v_b + eps)) * db

        trajectory.append((w, b))

    return trajectory
trajectory=do_adagrad()
plot_contour(trajectory=trajectory,label="ada_grad Path")