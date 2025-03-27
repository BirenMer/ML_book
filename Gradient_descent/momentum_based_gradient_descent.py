#WIP code
import numpy as np
#Sigmoid function f(x)

from trajectory_plotting_utils import plot_error_contour
from general_utils import data_points, error, grad_w,grad_b

def do_momentum_based_gradient_descent(init_w,init_b,max_epochs):
    X,Y=data_points()
    w,b,eta = init_w,init_b,1.0
    
    trajectory=[(w,b)]

    prev_v_w,prev_v_b,gamma=0,0,0.9
    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        
        w=w*v_w
        b=b*v_b

        trajectory.append((w,b))
        prev_v_w=v_w
        prev_v_b=v_b
        
        return trajectory

momentum_based_gradient_descent_trajectory=do_momentum_based_gradient_descent(-2,-2,1000)

plot_error_contour(error_function=error,trajectory=momentum_based_gradient_descent_trajectory,algo_name="Momentum Based GD")
