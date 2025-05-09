#vannila gradient descent
import numpy as np

from general_utils import data_points, grad_w,grad_b,error
from trajectory_plotting_utils import plot_contour



def do_gragient_descent(lr=0.1,max_epochs=1000):
    #trying to converge gradient descent for the below two point i.e. 
    # GD will optimized it self to reach convergence while trying to include this points in it trajectory. 
    # Data points
    X,Y = data_points()

    init_w = -2
    init_b = 2

    w,b = init_w,init_b
    eta=lr

    trajectory = [(w, b)]

    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
            
        # Average the gradients
        dw /= len(X)
        db /= len(X)

        # Parameter update
        w -= eta * dw
        b -= eta * db

        trajectory.append((w,b))

    return trajectory

gradient_descent_trajectory=do_gragient_descent()

plot_contour(gradient_descent_trajectory)