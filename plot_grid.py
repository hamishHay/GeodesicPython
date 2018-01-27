import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import sys
import ReadGrid

N = int(sys.argv[1])

Grid = ReadGrid.read_grid(N)

# GET NODES LIST
nodes = Grid.nodes

lats = Grid.lats
lons = Grid.lons

# lats = np.array(lats)
# lons = np.array(lons)
# friends = np.array(friends)
# data = np.cos(3*np.deg2rad(lats))*np.cos(np.deg2rad(lons))

# plt.scatter(lons,lats,c='b',s=0.5)
# plt.show()
# ax = plt.gca()
# ax.set_aspect('equal')

# # m = Basemap(projection='ortho',lon_0=-105,lat_0=40)
# m = Basemap(projection='hammer',lon_0=180)
m = Basemap(projection='hammer',lon_0=0,lat_0=90)
# x, y = m(lons,lats)
# m.scatter(x,y,marker='o',s=2,color='k')

for node in nodes:
# x, y = m(lons,lats)
    x, y = m(np.degrees(node.lon),np.degrees(node.lat))

    if abs(np.degrees(node.lat)) < 0.01:
        print("EQUATOR")
    m.scatter(x, y, marker='o',s=6,color='r')

    # friends = node.friends
    # num = 6
    # if friends[-1] < 0:
    #     num = 5
    #
    # if node.ID == 643:
    #     for j in range(num+1):
    #         for i in range(j):
    #             ind = friends[i]
    #             ind2 = friends[(i+1)%num]
    #             m.drawgreatcircle(np.degrees(lons[ind]),np.degrees(lats[ind]),np.degrees(lons[ind2]),np.degrees(lats[ind2]),c='k')
    #             print(ind)
    #             x, y = m(np.degrees(lons[ind]),np.degrees(lats[ind]))
    #             m.scatter(x, y, marker='o',s=6,color='k')
    #
    #             x, y = m(np.degrees(node.lon),np.degrees(node.lat))
    #             m.scatter(x, y, marker='o',s=6,color='r')
    #
    #             x, y = m(np.degrees(node.centroids[i][1]),np.degrees(node.centroids[i][0]))
    #             m.scatter(x, y, marker='o',s=6,color='r')
    #
    #         plt.show()

    # for i in range(num):
    #     ind = friends[i]
    #     ind2 = friends[(i+1)%num]
    #     m.drawgreatcircle(np.degrees(lons[ind]),np.degrees(lats[ind]),np.degrees(node.lon),np.degrees(node.lat),c='k',lw=0.8)
        # print(ind)
        # x, y = m(np.degrees(lons[ind]),np.degrees(lats[ind]))
        # m.scatter(x, y, marker='o',s=6,color='k')
        #
        # x, y = m(np.degrees(node.lon),np.degrees(node.lat))
        # m.scatter(x, y, marker='o',s=6,color='r')

# plt.show()

    #     # for k in range(3):
    #     #     mag += (b[num][k] - b[int(i)][k])**2
    #     # mag = np.sqrt(mag)
    #     if i >= 0:# and (x20 != 0 or x20 <350)  and (x2 != 0 or x20 <350):
    #         x2, y2 = m(lons[i],lats[i])
            # m.plot([x20,x2],[y20,y2],color='k')
            # if (lons[num] != 0 and lons[i] < 350) or (lons[i] != 0 and lons[num] < 350):
            # m.drawgreatcircle(lons[num],lats[num],lons[i],lats[i],c='k')
    # plt.show()
# m.pcolor(x,y,data,tri=True,cmap=plt.cm.hot)
# plt.savefig("COOOOL.png",bbox_inches='tight')
plt.show()
