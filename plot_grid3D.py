import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.basemap import Basemap
import sys
from netCDF4 import Dataset
import ReadGrid

N = int(sys.argv[1])

Grid = ReadGrid.read_grid(N)
nodes = Grid.nodes

fig = plt.figure(figsize=(12,12))
ax = Axes3D(fig)

lighting = False
cv = True

# data

# x = [0,1,1,0]
# y = [0,0,1,1]
# z = [0,1,0,1]
# verts = [list(zip(x, y,z))]
# ax.add_collection3d(Poly3DCollection(verts))
background_light = 0.12
light = np.array([0.0, -1e6, 0])
light_mag = np.sqrt(sum(light**2))

verts_added = []
for node in nodes:
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # print(node.ID, node.friends)
    f_num = 6
    if node.friends[-1] == -1:
        f_num = 5

    x = []#[node.x]
    y = []#[node.y]
    z = []#[node.z]

    n_vec = np.array([node.x, node.y, node.z])
    n_vec /= np.sqrt(sum(n_vec**2))
    if cv:
        for j in range(f_num):
            f_ID = node.friends[j]

            f_node = nodes[f_ID]

            centroid = node.centroids_cart[j]

            x.append(centroid[0])
            y.append(centroid[1])
            z.append(centroid[2])

            if lighting:
                bright = np.dot(light, n_vec)/(light_mag)
            else:
                bright = 1.0

            col = 1.0 * bright
            if col < 0:
                col = 0.0

            col += background_light
            if col > 1:
                col = 1.0

    else:
        for j in range(f_num):
            x = [node.x]
            y = [node.y]
            z = [node.z]

            f_ID1 = node.friends[j]
            f_ID2 = node.friends[(j+1)%f_num]

            f_node1 = nodes[f_ID1]
            f_node2 = nodes[f_ID2]

            x.append(f_node1.x)
            y.append(f_node1.y)
            z.append(f_node1.z)

            x.append(f_node2.x)
            y.append(f_node2.y)
            z.append(f_node2.z)

            if lighting:
                bright = (np.dot(light, n_vec)/(light_mag) + 0.6)/1.6
            else:
                bright = 1.0

            col = 1.0 * bright
            if col < 0:
                col = 0.0

            col += background_light
            if col > 1:
                col = 1.0

            # print(x,y,z)
            # cmap = plt.cm.hot
            # from random import random
            # col = cmap(col)
            # col = ()
            # col = cmap(random())
            verts = [list(zip(x, y,z))]
            curr_triang = list(np.sort([node.ID, f_ID1, f_ID2]))
            if len(verts_added) == 0:
                # ax.add_collection3d(Poly3DCollection(verts, linewidths=0.6, edgecolors=(0,0,0,1),facecolor=col))
                ax.add_collection3d(Poly3DCollection(verts, linewidths=0.6, edgecolors=(0.0,0.0,0.0,1.0),facecolor=(col,0,0,0.0)))
                verts_added.append(curr_triang)

            added = False
            for k in range(len(verts_added)):
                if curr_triang == verts_added[k]:
                    added = True

            if not added:
                print("plotting ", curr_triang)
                # ax.add_collection3d(Poly3DCollection(verts, linewidths=0.6, edgecolors=(0,0,0,1),facecolor=col))
                ax.add_collection3d(Poly3DCollection(verts, linewidths=0.6, edgecolors=(0.0,0.0,0.0,1.0),facecolor=(col,0,0,0.0)))

                verts_added.append(curr_triang)

    if cv:
        verts = [list(zip(x, y,z))]
        ax.add_collection3d(Poly3DCollection(verts, linewidths=0.3, edgecolors=(1*col,1*col,1*col,1.0),facecolor=(col,0,0,1.0)))

    # ax.set_aspect('equal')
    #
    # view_lim = 1.
    # ax.set_xlim([-view_lim,view_lim])
    # ax.set_ylim([-view_lim,view_lim])
    # ax.set_zlim([-view_lim,view_lim])
    #
    # ax.view_init(elev=20., azim=-60)
    #
    # # plt.axis('off')
    #
    # # savename = "/home/hamish/Dropbox/TAPS/grid_l" + str(N)
    # # if lighting:
    # #     savename += '_lighting'
    # # savename += ".pdf"
    # # plt.savefig(savename, bbox_inches='tight', transparent=True, facecolor=(0.05, 0.05, 0.05), edgecolor='none')
    # plt.show()

ax.set_aspect('equal')

view_lim = 1.
ax.set_xlim([-view_lim,view_lim])
ax.set_ylim([-view_lim,view_lim])
ax.set_zlim([-view_lim,view_lim])

ax.view_init(elev=20., azim=-60)

plt.axis('off')

savename = "/home/hamish/Dropbox/grid_l" + str(N)
if lighting:
    savename += '_lighting'
savename += ".pdf"
plt.savefig(savename, bbox_inches='tight') #  , transparent=True, facecolor=(0.05, 0.05, 0.05), edgecolor='none')
plt.show()
