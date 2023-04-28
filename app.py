#Gradient Descent to find values of m and b
# yhat = wx+b
#loss = (y-yhat)**2/N

#import packages
import pandas as pd
import numpy as np

#Initialise paramters
x = np.random.randn(10,1)
y = 1*x + np.random.rand()

#Setup paramaters(w & b)
w = 0
b = 0
#Setup hyperparamter
learning_rate = 0.01

#create function for gradient descent
def descent(x,y,learning_rate,w,b):
    dldw = 0
    dldb = 0
    N = x.shape[0]
    for xi,yi in zip(x,y):
        #loss = (y-(wx+b))**2
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
        
    #New parameters 
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w,b

#Iterate through epochs
for epoch in range(700):
    w,b = descent(x,y,learning_rate,w,b)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, paramters w:{w}, b:{b}')