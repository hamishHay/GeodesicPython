import numpy as np
from numpy import cos, sin, deg2rad, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.basemap import Basemap
import sys

class Node:
    def __init__(self, lat, lon, ID):
        self.lat = lat
        self.lon = lon

        self.coords_cart = self.sph2cart(np.pi*0.5 - lat, lon)
        self.x = self.coords_cart[0]
        self.y = self.coords_cart[1]
        self.z = self.coords_cart[2]

        self.pos_sph = np.array([lat, lon])
        self.ID = ID

        self.friends = []
        self.centroids = []
        self.centroids_cart = []

    def add_friend(self, ID):
        self.friends.append(ID)

    def add_centroid(self, lat, lon):
        self.centroids.append([lat, lon])
        self.centroids_cart.append(self.sph2cart(np.pi*0.5 - lat, lon))

    def create_polygon(self):
        f_num = 6
        centroids = self.centroids[:]
        if self.friends[-1] == -1:
            f_num = 5
            centroids = self.centroids[:-1]

        c_lats = []
        c_lons = []
        for i in range(f_num+1):
            c1_lat, c1_lon = np.degrees(self.centroids[i%f_num])

            c_lats.append(c1_lat)
            c_lons.append(c1_lon)

        self.polygon = SphericalPolygon.from_radec(c_lons, c_lats)

    def sph2map(self,lat1,lon1,lat2,lon2):
        m = 2.0 * (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
        x = m * np.cos(lat2) * np.sin(lon2 - lon1)
        y = m * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

        if abs(x) < 1e-6: x = 0.0
        if abs(y) < 1e-6: y = 0.0

        return np.array([x, y])

    def sph2cart(self, theta, phi, r=1.0):
        theta = theta
        phi = phi

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        # coords_cart = np.array([x, y, z])

        return np.array([x, y, z])


class Shape:
    def __init__(self, nodes, N, npole_node):
        self.nodes = nodes
        self.node_num = N
        self.npole_node = npole_node

        self.lats = []
        self.lons = []
        for i in range(len(nodes)):
            # print(i, len(nodes))
            self.lats.append(nodes[i].lat)
            self.lons.append(nodes[i].lon)

    def reorder(self,perms):
        new_list = []
        for i in range(len(self.nodes)):
            new_list.append(self.nodes[perms[i]])
            new_list[i].ID = i

        # print([i.lat for i in new_list])
        for i in range(len(self.nodes)):
            old_friends = new_list[i].friends
            new_list[i].friends = []

            pent = 0
            if old_friends[-1] < 0: pent = 1
            for j in range(len(old_friends)-pent):
                old_ID = old_friends[j]
                new_ID = self.nodes[old_ID].ID#perms.index(old_ID)

                new_list[i].friends.append(new_ID)

            if pent:
                new_list[i].friends.append(-1)



        self.nodes = new_list


        self.lats = []
        self.lons = []
        for i in range(len(self.nodes)):
            self.lats.append(self.nodes[i].lat)
            self.lons.append(self.nodes[i].lon)

            # print(np.rad2deg(self.lats[i]), np.rad2deg(self.nodes[i].lat))


    def get_friends_list(self, ID, level=0):
        def get_node_friends(node):
            friends_list = []
            for f in node:
                friends_list.extend(self.nodes[f].friends)
            return friends_list

        friends_list = []
        new_friends = get_node_friends([ID])
        if new_friends[-1] == -1:
            new_friends = new_friends[:-1]

        friends_list.extend(new_friends)
        for i in range(level+1):
            new_friends = get_node_friends(friends_list)
            friends_list.extend(new_friends)
            friends_list = list(set(friends_list))

        return friends_list

    def save_grid(self, overwrite=True):
        file_name = "grid_l" + str(self.node_num) + ".txt"
        if not overwrite: file_name += "_reordered"

        outFile = open(file_name, "w");

        outFile.write("{:5} {:12} {:12} {:38} {:20}\n".format("ID", "NODE LAT", "NODE LON", "FRIENDS LIST", "CENTROID COORD LIST"));

        print(len(self.nodes))
        for i in range(len(self.nodes)):
            lat = np.rad2deg(self.lats[i])
            lon = np.rad2deg(self.lons[i])
            f = self.nodes[i].friends
            centroids = np.rad2deg(np.array(self.nodes[i].centroids))
            cx = centroids[:,0]
            cy = centroids[:,1]

            outFile.write("{:5d} {:12.16f} {:12.16f} {{ {:4d}, {:4d}, {:4d}, {:4d}, {:4d}, {:4d}}}, {{( {:10.16f}, {:10.16f}), ( {:10.16f}, {:10.16f}), ( {:10.16f}, {:10.16f}), ( {:10.16f}, {:10.16f}), ( {:10.16f}, {:10.16f}), ( {:10.16f}, {:10.16f})}} \n".format(self.nodes[i].ID, lat, lon, f[0], f[1], f[2], f[3], f[4], f[5], cx[0], cy[0], cx[1], cy[1], cx[2], cy[2], cx[3], cy[3], cx[4], cy[4], cx[5], cy[5]));



        outFile.close()

def read_grid(N):

    load_file = '/home/hamish/Research/GeodesicPython/grid_l' + str(N) + '.txt'
    # load_file = 'cpp_version/test_file.txt'

    print("\nReading grid from file", load_file)

    f = open(load_file,'r')
    lines = f.readlines()

    node_num = len(lines)-1#10 * ((2**(N-1))**2) + 2
    # Strange notation creates a list of the desired length
    node_list = [None] * node_num

    npole_node = None

    line_count = 0.0

    for line in lines[1:]:
    # for line in lines:
        line = line.split()

        line_count += 1.0

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

        outstring = "\t Percentage complete... {:.1f} %".format(line_count/float(node_num)*100.0)

        sys.stdout.write(outstring)
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.flush()


    f.close()

    print("\n\t Finished!")

    GeodGrid = Shape(node_list, N, npole_node)

    return GeodGrid
