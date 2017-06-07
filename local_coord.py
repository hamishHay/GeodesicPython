import numpy as np
from numpy import cos, sin, deg2rad, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

a = 252.1

def mapping(lat0, lon0, lat1, lon1):

    m = 2.0 / (1 + sin(lat1)*sin(lat0) + cos(lat1)*cos(lat0)*cos(lon1-lon0))

    return m


def x_coord(lat0, lon0, lat1, lon1):

    m = mapping(lat0, lon0, lat1, lon1)

    x = m * a * cos(lat1) * sin(lon1 - lon0)

    return x

def y_coord(lat0, lon0, lat1, lon1):

    m = mapping(lat0, lon0, lat1, lon1)

    y = m*a*(sin(lat1)*cos(lat0) - cos(lat1)*sin(lat0)*cos(lon1-lon0))

    return y

load_file = 'grid_l1_testing.txt'

f = open(load_file,'r')
lines = f.readlines()

# [print(i) for i in lines]

lats = []
lons = []
friends = []
for line in lines:
    all = []
    line = line.split()


    lats.append(float(line[1]))
    lons.append(float(line[2]))

    if line[3] == "{":
        line.remove("{")

    start = 3
    for i in range(6):
        line[start+i] = line[start+i].strip('{')
        line[start+i] = line[start+i].strip('}')
        line[start+i] = line[start+i].strip(',')

        all.append(int(line[start+i]))
    friends.append(all)
f.close()

lats = deg2rad(np.array(lats))
lons = deg2rad(np.array(lons))
friends = np.array(friends)

for t in range(len(friends)):
    lat1 = lats[t]
    lon1 = lons[t]

    xs = [x_coord(lat1,lon1,lat1,lon1)]
    ys = [y_coord(lat1,lon1,lat1,lon1)]

    if friends[t][-1] != -1:
        length = 6
    else:
        length = 5

    for i in range(length):
        lat2 = lats[friends[t][i]]
        lon2 = lons[friends[t][i]]

        x = x_coord(lat1,lon1,lat2,lon2)
        y = y_coord(lat1,lon1,lat2,lon2)

        xs.append(x)
        ys.append(y)

    print(xs,ys)
    plt.scatter(xs,ys)
        # plt.show()

    plt.grid()
    plt.show()

