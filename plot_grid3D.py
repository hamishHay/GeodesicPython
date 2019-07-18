import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.basemap import Basemap
import sys
import h5py
import ReadGrid

cmap = mpl.cm.get_cmap('plasma')

N = int(sys.argv[1])

Grid = ReadGrid.read_grid(N)
nodes = Grid.nodes

fig = plt.figure(figsize=(12,12))
ax = Axes3D(fig)

lighting = False
cv = False

# in_file = h5py.File("../GeodesicODISBeta/GeodesicODIS/flux/DATA_OBL/data.h5", 'r')

start_f = -400

background_light = 0.12
light = np.array([0.0, -1e6, 0])
light_mag = np.sqrt(sum(light**2))

verts_added = []
poly_collection = []
st = 40
# for node in nodes[:]:
    # try:
    #     node.friends.remove(-1)
    # except ValueError:
    #     continue


for node in nodes[:]:#[st:st+1]:

    # print(node.ID)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # print(node.ID, node.friends)
    f_num = len([y for y in node.friends if y >= 0])
    # print(node.friends, f_num)
    face_num = f_num
    boundary = False
    if node.friends.count(-2) > 0:
        f_num -= 1
        face_num = f_num - node.friends.count(-2)
        f_num_b = 6 - node.friends.count(-1) - node.friends.count(-2)
        boundary = True

    # if node.friends[-1] == -1:
    #     f_num = 5

    x = []#[node.x]
    y = []#[node.y]
    z = []#[node.z]

    n_vec = np.array([node.x, node.y, node.z])
    n_vec /= np.sqrt(sum(n_vec**2))

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

    if cv:
        if boundary: cv_num = f_num_b - 1
        else: cv_num = f_num
        for j in range(cv_num):
            f_ID = node.friends[j]

            if f_ID >= 0:
                f_node = nodes[f_ID]

                centroid = node.centroids_cart[j]
                # centroid = f_node.coords_cart

                x.append(centroid[0])
                y.append(centroid[1])
                z.append(centroid[2])

        if boundary:
            # find first friend
            f_ID = node.friends[0]
            f_node = nodes[f_ID]
            edge_coord = (node.coords_cart + f_node.coords_cart)*0.5

            # add midpoint between node and fist friend
            x.insert(0, edge_coord[0])
            y.insert(0, edge_coord[1])
            z.insert(0, edge_coord[2])

            # find last friend
            indx = next(a for a, val in enumerate(node.friends) if val < -1) - 1

            f_ID = node.friends[indx]
            f_node = nodes[f_ID]
            edge_coord = (node.coords_cart + f_node.coords_cart)*0.5

            # add midpoint between node and last friend
            x.append(edge_coord[0])
            y.append(edge_coord[1])
            z.append(edge_coord[2])

            # add node
            x.append(node.coords_cart[0])
            y.append(node.coords_cart[1])
            z.append(node.coords_cart[2])

                # print(x, y, z)



        # col = cmap(data_x[data_c][node.ID])

    else:
        new_f_list = [k for k in node.friends if k >= 0]

        print(new_f_list)
        # fig,ax11 = plt.subplots()
        for j in range(face_num):
            x = [node.x]
            y = [node.y]
            z = [node.z]

            # f_ID1 = node.friends[j]
            # f_ID2 = node.friends[(j+1)%f_num]
            f_ID1 = new_f_list[j]
            f_ID2 = new_f_list[(j+1)%f_num]

            # print(node.ID, f_ID1, f_ID2)

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

            ec = 0.1

            verts = [list(zip(x, y,z))]
            curr_triang = list(np.sort([node.ID, f_ID1, f_ID2]))
            if len(verts_added) == 0:
                poly = Poly3DCollection(verts, linewidths=0.05, edgecolors=(ec,ec,ec,1.0),facecolor=(col,0,0,0.8))
                ax.add_collection3d(poly)
                poly_collection.append(poly)
                verts_added.append(curr_triang)

            added = False
            for k in range(len(verts_added)):
                if curr_triang == verts_added[k]:
                    added = True

            if not added:
                # print("plotting ", curr_triang)
                # ax.add_collection3d(Poly3DCollection(verts, linewidths=0.6, edgecolors=(0,0,0,1),facecolor=col))
                poly = Poly3DCollection(verts, linewidths=0.05, edgecolors=(ec,ec,ec,1.0),facecolor=(col,0,0,0.8))
                ax.add_collection3d(poly)
                poly_collection.append(poly)

                verts_added.append(curr_triang)
    # plt.show()
    if cv:
        verts = [list(zip(x, y,z))]
        # print(verts)
        # if f_num == 5: ax.add_collection3d(Poly3DCollection(verts, linewidths=1.4, edgecolors=(1*col,1*col,1*col,1.0),facecolor=(col*0.5,0,0,1.0)))
        # else: ax.add_collection3d(Poly3DCollection(verts, linewidths=1.4, edgecolors=(1*col,1*col,1*col,1.0),facecolor=(col,0,0,1.0)))
        ec = 0.1
        poly = Poly3DCollection(verts, linewidths=0.05, edgecolors=(ec,ec,ec,1.0),facecolor=(col,0,0,0.8))
        poly_collection.append(poly)
        ax.add_collection3d(poly)

ax.set_aspect('equal')

view_lim = 1.
ax.set_xlim([-view_lim,view_lim])
ax.set_ylim([-view_lim,view_lim])
ax.set_zlim([-view_lim,view_lim])

ax.view_init(elev=0., azim=0)

plt.axis('off')

savename = "/home/hamish/Dropbox/grid_l" + str(N)
# savename = "art/grid_l" + str(N)
if lighting:
    savename += '_lighting'
savename += ".pdf"
plt.savefig(savename, dpi=300)
# count = 0
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
#     savename = "/home/hamish/Dropbox/Spinning/grid_l" + str(N)+'_'+str(count)+'.png'
#     count+=1
#     plt.savefig(savename, dpi=100, bbox_inches='tight', transparent=True)
#     for i in range(len(poly_collection)):
#         poly_collection[i].set_facecolor(cmap(data_x[count][i]))
# plt.show()
