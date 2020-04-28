import matplotlib as mpl
mpl.use('Agg')
import ReadGrid
import numpy as np
from numpy import deg2rad
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
# from shapely.geometry import Polygon

def main():
    N = int(sys.argv[1])        # geodesic grid recursion level


    r = 1560.8e3                # spherical radius

    # Load geodesic grid corresponding to level N
    Grid = ReadGrid.read_grid(N)

    # Get a list of and number of nodes in the geodesic grid
    nodes = Grid.nodes
    gg_node_num = len(nodes)

    test = np.zeros(gg_node_num)

    for i in range(len(nodes)):
        node_main = nodes[i]

        lat0, lon0 = [node_main.lat, node_main.lon]

        test[i] = np.cos(lat0)**2.0 + (1 + np.sin(2*lon0))


    nodeOperators = np.zeros((gg_node_num, 6, 7))

    for i in range(len(nodes)):
        node_main = nodes[i]

        f_num = len(node_main.friends)
        if node_main.friends[-1] < 0:
            f_num -= 1

        V = np.zeros((f_num+1, 6))



        lat0, lon0 = [node_main.lat, node_main.lon]

        s = np.zeros(f_num+1)
        s[0] = test[i]

        V[:, 0] = 1.0
        for j in range(f_num):
            f_ID = node_main.friends[j]

            node_f = nodes[f_ID]
            s[j+1] = test[f_ID]

            lat1, lon1 = [node_f.lat, node_f.lon]



            xf, yf = sph2map(lat0, lon0, lat1, lon1, r)


            V[j+1, 0+1] = xf
            V[j+1, 1+1] = yf
            V[j+1, 2+1] = xf**2.0
            V[j+1, 3+1] = xf*yf
            V[j+1, 4+1] = yf**2.0

         
        # Get least squares operator using the pseudo-inverse 
        interpOperator = np.matmul(np.linalg.pinv(np.matmul(V.T, V)), V.T) 

        # print(interpOperator.shape)
        if f_num == 5:
            nodeOperators[i, :, :-1] = interpOperator
        else:
            nodeOperators[i] = interpOperator 

        c = np.matmul(interpOperator, s)

        if i==0:
            print(interpOperator)

        # print(c[0])

    
    # print(nodeOperators[0])
        # print(test[f_ID], xf*c[0] + yf*c[1] + xf**2.0*c[2] + xf*yf*c[3] + yf**2.0*c[4])
        print(test[f_ID], c[0] + xf*c[0+1] + yf*c[1+1] + xf**2.0*c[2+1] + xf*yf*c[3+1] + yf**2.0*c[4+1])
    SaveMatrix(N, nodeOperators)




def sph2map(lat1,lon1,lat2,lon2, r):
    """
    Find the projected xy coordinates of the point latlon2 around latlon1 using
    a stereographic projection. The inverse of this function is map2sph.
    """

    m = 2.0 / (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    x = m * r * np.cos(lat2) * np.sin(lon2 - lon1)
    y = m * r * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

    return np.array([x, y])




def map2sph(lat1, lon1, x, y, r, trig=False):
    """
    Find the latlon coordinates from the projected coordinates xy about the point
    latlon1 using a stereographic projection. This function is the inverse of
    sph2map.
    """

    rho = np.sqrt(x**2. + y**2.)
    c = 2. *np.arctan2(rho, 2.*r)

    sinLat = np.cos(c)*np.sin(lat1) + y*np.sin(c)*np.cos(lat1)/rho
    lat = np.arcsin(sinLat)

    lon = lon1 + np.arctan2(x*np.sin(c), (rho*np.cos(lat1)*np.cos(c) - y*np.sin(lat1)*np.sin(c)))
    if not trig:
        return np.array([lat, lon])
    else:
        return np.array([lat, lon, sinLat])





def SaveMatrix(N, data):
    """
    Save the least-squares matrices to an hdf5 file.
    """

    file_name = "grid_l" + str(N) + "_" + "least_squares_operator.h5"
    print("Saving least-squares matrices to " + file_name)

    f = h5py.File(file_name, 'w')

    dset_data = f.create_dataset("operator", data=data, dtype='d')

    f.close()


if __name__=='__main__':
    main()
