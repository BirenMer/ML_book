
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


def do_rmsprop(init_w,init_b,max_epochs):
    w,b,eta=init_w,init_b,0.1
    v_w,v_b,eps=0,0,1e-8
    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
        v_w=v_w+dw**2
        v_b=v_b+db**2

        w=w-(eta/np.sqrt(v_w+eps))*dw
        b=b-(eta/np.sqrt(v_b+eps))*db
        