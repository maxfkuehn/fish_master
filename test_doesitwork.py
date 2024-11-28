import numpy as np
from IPython import embed

x1=8
x0=4
y0=2
y1=7

X = np.random.rand(10)*10
Y = np.random.rand(10)*10

indexX = np.where(np.logical_and(X <= x1, X >= x0))

index = np.where((Y[indexX] <= y1) & (Y[indexX] >= y0))



a= X[index[0]]
b= Y[index[0]]
embed()
