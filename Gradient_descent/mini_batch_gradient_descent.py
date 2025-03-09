#WIP code
import numpy as np
#Sigmoid function f(x)

#datapoints
X=[0.5,2.5]
Y=[0.2,0.0]


def f(w,b,x):
    return 1.0/(1.0 * np.exp(-(w*x+b)))


#Error function
def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx=f(w,b,x)
        err+=0.5*(fx-y)**2
    return err


def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y) * (fx) * (1-fx)

def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y) * (fx) * (1-fx) * x

#this code updates the parameters at evey point,
def do_mini_batch_gragient_descent():
    
    w,b,eta,max_epochs=-2,-2,1.0,1000
    mini_batch_size,num_points_seen=2,0
    
    for i in range(max_epochs):
        dw,db,num_points=0,0,0
        
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
            num_points_seen+=1

            if num_points_seen%mini_batch_size==0:
            #updating parameters after we have seen a batch of parameters (unlike stochastic GD i.e. insted of updating them at each and evey data point )
                w=w-eta*dw
                b=b-eta*db
        # print(w,b)