import numpy as np
import matplotlib.pyplot as plt
import Grid
from Node import Node
# from timer import Timer
import sys

def main():
    grid = Grid.Grid()

    r = 1.0

    [x1, y1, z1] = sph2cart(r, 0.0, 0.)
    n1 = Node(x1, y1, z1, 0)
    n1.f_num = 5

    [x2, y2, z2] = sph2cart(r, 180.0, 0.0)
    n2 = Node(x2, y2, z2, 1)
    n2.f_num = 5

    [x3, y3, z3] = sph2cart(r,90.0-np.arctan(0.5)*180./np.pi, 36.0)
    n3 = Node(x3, y3, z3, 2)
    n3.f_num = 5

    [x4, y4, z4] = sph2cart(r,90.0-np.arctan(0.5)*180./np.pi, 36.0*3.0)
    n4 = Node(x4, y4, z4, 3)
    n4.f_num = 5

    [x5, y5, z5] = sph2cart(r,90.0-np.arctan(0.5)*180./np.pi, 36.0*5.0)
    n5 = Node(x5, y5, z5, 4)
    n5.f_num = 5

    [x6, y6, z6] = sph2cart(r,90.0-np.arctan(0.5)*180./np.pi, 36.0*7.0)
    n6 = Node(x6, y6, z6, 5)
    n6.f_num = 5

    [x7, y7, z7] = sph2cart(r,90.0-np.arctan(0.5)*180./np.pi, 36.0*9.0)
    n7 = Node(x7, y7, z7, 6)
    n7.f_num = 5

    [x8, y8, z8] = sph2cart(r,(90.0+np.arctan(0.5)*180./np.pi), 0.0)
    n8 = Node(x8, y8, z8, 7)
    n8.f_num = 5

    [x9, y9, z9] = sph2cart(r,(90.0+np.arctan(0.5)*180./np.pi), 36.0*2.0)
    n9 = Node(x9, y9, z9, 8)
    n9.f_num = 5

    [x10, y10, z10] = sph2cart(r,(90.0+np.arctan(0.5)*180./np.pi), 36.0*4.0)
    n10 = Node(x10, y10, z10, 9)
    n10.f_num = 5

    [x11, y11, z11] = sph2cart(r,(90.0+np.arctan(0.5)*180./np.pi), 36.0*6.0)
    n11 = Node(x11, y11, z11, 10)
    n11.f_num = 5

    [x12, y12, z12] = sph2cart(r,(90.0+np.arctan(0.5)*180./np.pi), 36.0*8.0)
    n12 = Node(x12, y12, z12, 11)
    n12.f_num = 5

    grid.add_node(n1)
    grid.add_node(n2)
    grid.add_node(n3)
    grid.add_node(n4)

    grid.add_node(n5)
    grid.add_node(n6)
    grid.add_node(n7)
    grid.add_node(n8)

    grid.add_node(n9)
    grid.add_node(n10)
    grid.add_node(n11)
    grid.add_node(n12)

    grid.find_friends()

    N = int(sys.argv[1])
    for i in range(N-1):
        print("Processing G" + str(i+2))
        grid.bisect_grid()

    #     print("\t " + str(len(grid.node_list)) + " points.")
    #
    grid.create_master_friends_list()

    grid.order_friends()
    # grid.spring_dynamics(N)
    grid.find_centers()
    # grid.shift_centroids()
    #
    grid.save_grid('grid_l'+str(N)+'_test.txt')

def sph2cart(r,theta,phi):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return np.array([x, y, z])

def cart2sph(coords):
    x = coords[0]
    y = coords[1]
    z = coords[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.pi*0.5 - np.arccos(z/r)
    lon = np.arctan2(y,x)


    if np.rad2deg(lon)+180.0 > 359.9:
        return np.array([r, np.rad2deg(lat), 0.0])
    return np.array([r, np.rad2deg(lat), np.rad2deg(lon)+180.0])

 # you can omit in most cases as the destructor will call it

if __name__ == '__main__':
    main()
