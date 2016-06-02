# 2x+3y-z =  1

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy.random import *
from mpl_toolkits.mplot3d import Axes3D

N =10

#xdata = np.array([0,1,2,3,4,5,6])
#xdata = np.arange(0.0,10.0,1.0)

np.random.seed(0)

xdata = 5*randn(N)
#ydata = np.array([0.1,0.9,2.2,2.8,4.2,5.9,7.4])
ydata =  5*randn(N)
zdata = 2*xdata + 3*ydata -1 + 10*randn(N) 
zdata[N-1] = 100

fig = plt.figure()
ax = Axes3D(fig)


A = np.ones(N)
A =np.vstack(([A], [xdata],[ydata])) 
#A = A.T
#w = np.array([1,2])
D = A.dot(A.T)
lamb = 10
I = np.identity(3) 
AA = np.linalg.inv(D+lamb*I)
AA = AA
#print AA
d = A.dot(zdata)
w = AA.dot(d)
x = np.linspace(-20,20,10)
y = np.linspace(-20,20,10)
X, Y = np.meshgrid(x,y)
z = w[0] +X*w[1] + Y*w[2]

#z = -1 + 2*X + 3*Y

plt.plot(xdata,ydata,zdata, 'ro')
#plt.plot(x,y,z,"r-")
ax.plot_surface(X,Y,z,rstride = 1,cstride=1)
#ax.plot3D(X,Y,z)


plt.show()
