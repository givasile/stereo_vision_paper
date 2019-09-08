import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# configuration
start = -5
stop = 5
num = 1000

# image
# X = np.linspace(start, stop, num)
# Y = np.linspace(start, stop, num)
# tmp = np.meshgrid(X,Y)

# tmp1 = np.abs(tmp[0]+tmp[1])/2
# tmp2 = np.exp(np.abs(tmp[0]-tmp[1]))
# Z = tmp1 * tmp2

# plt.figure()
# plt.imshow(Z, cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show(block = False)


# 3D plot
X = np.outer(np.linspace(start, stop, num), np.ones(num))
Y = X.copy().T
tmp1 = np.abs(X + Y)/2
tmp2 = np.exp(np.abs(X-Y))
Z = tmp1 * tmp2

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z, cmap = plt.cm.jet)
plt.show(block = False)
