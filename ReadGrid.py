import numpy as np
from numpy import cos, sin, deg2rad, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import sys

class Node:
    def __init__(self, lat, lon, ID):
        self.lat = lat
        self.lon = lon
        self.pos_sph = np.array([lat, lon])
        self.ID = ID

        self.friends = []
        self.centroids = []

    def add_friend(self, ID):
        self.friends.append(ID)

    def add_centroid(self, lat, lon):
        self.centroids.append(np.array([lat, lon]))

        # if len(self.centroids) == 6:
        #     # print(self.centroids)
        #     f_num = 6
        #     if self.friends[-1] < 0:
        #         f_num = 5
        #
        #     lat1 = self.lat
        #     lon1 = self.lon
        #
        #     x_min = 0
        #     x_max = 0
        #     for i in range(f_num):
        #         x, y = self.sph2map(lat1,lon1,self.centroids[i][0],self.centroids[i][1])
        #         x_min = min(x_min, x)
        #         x_max = max(x_max, x)
        #
        #     self.centroid_width = np.sqrt((x_max-x_min)**2.0)

    def sph2map(self,lat1,lon1,lat2,lon2):
        m = 2.0 * (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
        x = m * np.cos(lat2) * np.sin(lon2 - lon1)
        y = m * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

        if abs(x) < 1e-6: x = 0.0
        if abs(y) < 1e-6: y = 0.0

        return np.array([x, y])


class Shape:
    def __init__(self, nodes, N, npole_node):
        self.nodes = nodes
        self.node_num = N
        self.npole_node = npole_node

        self.lats = []
        self.lons = []
        for i in range(len(nodes)):
            self.lats.append(nodes[i].lat)
            self.lons.append(nodes[i].lon)

# N = int(sys.argv[1])

def read_grid(N):

    load_file = '/home/hamish/Research/GeodesicODIS/input_files/grid_l' + str(N) + '.txt'

    f = open(load_file,'r')
    lines = f.readlines()

    node_num = 10 * ((2**(N-1))**2) + 2

    # Strange notation creates a list of the desired length
    node_list = [None] * node_num

    npole_node = None

    for line in lines[1:]:
        line = line.split()

        ID = int(line[0])
        lat = deg2rad(float(line[1]))
        lon = deg2rad(float(line[2]))
        del line[:3]

        # node_list.append( Node(lat, lon, ID) )
        node_list[ID] = Node(lat, lon, ID)
        if lat > deg2rad(89.999): npole_node = node_list[ID]

        if '{' in line: line.remove('{')
        for i in range(6):
            if line[i][0] == '{': line[i] = line[i][1:]
            if line[i].endswith(','): line[i] = line[i][:-1]
            if line[i].endswith('}'): line[i] = line[i][:-1]

            node_list[ID].add_friend(int(line[i]))

        del line[:6]

        if line[0] == '{(': del line[0]
        elif line[0][:2] == '{(': line[0] = line[0][2:]

        for i in range(6):
            coord = np.zeros(2)
            for j in range(2):
                if line[j].endswith(','): line[j] = line[j][:-1]
                if line[j].endswith(')'): line[j] = line[j][:-1]
                if line[j].endswith(')}'): line[j] = line[j][:-2]

                coord[j] = deg2rad(float(line[j]))

            node_list[ID].add_centroid(coord[0], coord[1])
            del line[:2]

            try:
                if line[0] == '(': del line[0]
                elif line[0][0] == '(': line[0] = line[0][1:]

            except IndexError:
                pass

    f.close()

    GeodGrid = Shape(node_list, N, npole_node)

    return GeodGrid
