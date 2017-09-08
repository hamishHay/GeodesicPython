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


load_file = 'grid_l2.txt'

f = open(load_file,'r')
lines = f.readlines()

node_list = []

for line in lines[1:]:
    line = line.split()

    ID = int(line[0])
    lat = deg2rad(float(line[1]))
    lon = deg2rad(float(line[2]))
    del line[:3]

    node_list.append(Node(lat, lon, ID))

    if '{' in line: line.remove('{')
    for i in range(6):
        if line[i].endswith(','): line[i] = line[i][:-1]
        if line[i].endswith('}'): line[i] = line[i][:-1]

        node_list[-1].add_friend(int(line[i]))

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

        node_list[-1].add_centroid(coord[0], coord[1])
        del line[:2]

        try:
            if line[0] == '(': del line[0]
            elif line[0][0] == '(': line[0] = line[0][1:]

        except IndexError:
            pass

    print(line)

f.close()
