import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

def fun(x, y):
    return 222 * np.square(x) + 0.5 * 44 * np.square(y)

fig = plt.figure()
x = np.arange(0, 1, 0.01)
y = np.arange(0, np.pi, 0.01)
X, Y = np.meshgrid(x, y)
zs = np.array(fun(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

ax = fig.gca(projection='3d')
ax.set_xlabel('distance')
ax.set_ylabel('angle')
ax.set_zlabel('potential (CCC1)')
ax.plot_surface(X, Y, Z)
ax.legend()

plt.show()