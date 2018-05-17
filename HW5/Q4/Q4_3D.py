import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

X = np.array([
    [1,  1],
    [1, -1],
    [-1, 1],
    [-1,-1],
    [2,  2],
    [2, -2],
    [-2, 2],
    [-2, -2]
])

Y = np.array([1,1,1,1,-1, -1, -1, -1])

X_pro = np.zeros((X.shape[0], 3))
X_pro[:,0] = X[:,0]**2
X_pro[:,1] = np.sqrt(2)*X[:,1]*X[:,0]
X_pro[:,2] = X[:,1]**2
X = X_pro
print "Projected data in 3D\n",X
# Fit the data with an svm
svc = SVC(kernel='linear')
svc.fit(X,Y)

# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should for the last coordinate to plot
# the plane in terms of x and y.

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]) / svc.coef_[0][2]
tmp = np.linspace(-4,4,50)
x,y = np.meshgrid(tmp,tmp)

# Plot stuff.
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z(x,y))
ax.plot3D(X[Y==-1,0], X[Y==-1,1], X[Y==-1,2],'sy')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
plt.show()