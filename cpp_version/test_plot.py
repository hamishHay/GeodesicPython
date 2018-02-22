import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

vs1 = np.loadtxt("xyz_n12.txt").T
vs2 = np.loadtxt("xyz.txt").T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# n1 = -np.cross(v4,v5)
# n2 = -np.cross(v5,v6)
# n3 = -np.cross(v6,v4)
#
# print(np.dot(v, n1), np.dot(v, n2), np.dot(v, n3))

# vs = np.array([v4, v5, v6]).T

ax.scatter(vs1[0], vs1[1], vs1[2], c='r')
ax.scatter(vs2[0], vs2[1], vs2[2], c='b')
# ax.scatter(v[0], v[1], v[2], 'r')

view_lim = 1.
ax.set_xlim([-view_lim,view_lim])
ax.set_ylim([-view_lim,view_lim])
ax.set_zlim([-view_lim,view_lim])

plt.show()
