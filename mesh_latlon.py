import ReadGrid
import numpy as np
from numpy import deg2rad
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import interpolate

def does_intersect(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

    if (s >= 0 and s <= 1 and t >= 0 and t <= 1):
        return 1

    return 0

def sph2map(lat1,lon1,lat2,lon2):
    m = 2.0 * (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    x = m * np.cos(lat2) * np.sin(lon2 - lon1)
    y = m * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

    if abs(x) < 1e-6: x = 0.0
    if abs(y) < 1e-6: y = 0.0

    return np.array([x, y])

def distanceBetween(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def getVandermondeMatrix(node_list, node_ID, inv=False):
    f_num = 6
    node = node_list[node_ID]
    if node.friends[-1] < 0:
        f_num = 5

    V = np.ones((6,6), dtype=np.float64)

    lat1 = node.lat
    lon1 = node.lon
    for i in range(f_num):
        f1_ID = node.friends[i]

        f1 = node_list[f1_ID]

        lat2 = f1.lat
        lon2 = f1.lon

        x, y = sph2map(lat1, lon1, lat2, lon2)

        V[i,0] = 1.0
        V[i,1] = x
        V[i,2] = x**2.0
        V[i,3] = y
        V[i,4] = x*y
        V[i,5] = y**2.0

    if f_num == 5:
        V[-1,1:] = 0.0

    U, L, VT = np.linalg.svd(V)

    V = VT.T
    UT = U.T

    L_inv = np.diag(1.0/L)

    L_inv[abs(L_inv) > 4e1] = 0.0

    A_inv = np.linalg.multi_dot([V, L_inv, UT]) #V.dot(L_inv).dot(U)

    if inv:
        return A_inv

    return V

def test_interp(data_file, sl, grid, ll_lat, ll_lon, ll_grid_link, V_inv):
    data = h5py.File(data_file, 'r')

    n_field = np.array(data["displacement"][sl])

    gg_lat = np.array(grid.lats)
    gg_lon = np.array(grid.lons)

    triang = tri.Triangulation(gg_lon, gg_lat)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,3),dpi=120)
    tcnt = ax1.tricontourf(triang, n_field/1e3, 21, cmap = plt.cm.coolwarm)
    cb = plt.colorbar(tcnt, ax=ax1, orientation='horizontal')
    ax1.set_title("ODIS DATA")

    nodes = grid.nodes

    data_interp = np.zeros((len(ll_lon), len(ll_lat)))
    for i in range(len(ll_lon)):
        for j in range(len(ll_lat)):
            # SET DATA VECTOR
            d = np.zeros((6,1))

            cell_ID = ll_grid_link[i,j]

            f_num = 6
            if nodes[cell_ID].friends[-1] < 0:
                f_num = 5
                d[-1,0] = n_field[cell_ID]

            # SET DATA VECTOR FROM IMPORTED NUMERICAL FIELD
            for k in range(f_num):
                f_ID = nodes[cell_ID].friends[k]
                d[k,0] = n_field[f_ID]

            # FIND LAT-LON GRID MAPPING COORDS
            lat1 = nodes[cell_ID].lat
            lon1 = nodes[cell_ID].lon

            lat2 = ll_lat[j]
            lon2 = ll_lon[i]

            x, y = sph2map(lat1, lon1, lat2, lon2)

            # SET INVERSE VANDERMONDE MATRIX
            VAND_INV = V_inv[cell_ID]

            # CALCULATE C COEFFS
            c = VAND_INV.dot(d)
            c = c[:, 0]

            # EXPAND FUNCTION TO FIND INTERPOLATED DATA
            data_interp[i,j] = c[0] + c[1]*x + c[2]*x**2.0 + c[3]*y + c[4]*y*x + c[5]*y**2.0


    data_interp = data_interp.T / 1e3

    tcnt2 = ax2.contourf(ll_lon, ll_lat, data_interp, 21, cmap = plt.cm.coolwarm)
    cb2 = plt.colorbar(tcnt2, ax=ax2, orientation='horizontal', label="Pressure [kPa]")
    ax2.set_title("My interpolation")

    gg_lat = np.pi*0.5 - gg_lat
    knotst, knotsp = gg_lat.copy(), gg_lon.copy()

    f = interpolate.SmoothSphereBivariateSpline(gg_lat, gg_lon, n_field/np.amax(n_field), s=1.5)

    data_interp2 = f(np.pi*0.5- ll_lat, ll_lon) * np.amax(n_field) / 1e3
    tcnt3 = ax3.contourf(ll_lon, ll_lat, data_interp2, 21, cmap = plt.cm.coolwarm)
    cb3 = plt.colorbar(tcnt3, ax=ax3, orientation='horizontal')
    ax3.set_title("SciPy Spline Interpolation")

    n_field /= 1e3
    print("DATA INFO: MAX:", np.amax(n_field), "MIN:", np.amin(n_field))
    print("INTP INFO: MAX:", np.amax(data_interp), "MIN:", np.amin(data_interp),
          "%:", (np.amax(n_field)-np.amax(data_interp))/np.amax(n_field)*100,
          (np.amin(n_field)-np.amin(data_interp))/np.amin(n_field)*100)
    print("INTP2 INFO: MAX:", np.amax(data_interp2), "MIN:", np.amin(data_interp2))

    plt.show()

def SaveVandermonde2HDF5(N, dx, node_list, V_inv, cell_pos):
    f = h5py.File("grid_l" + str(N) + "_" + str(int(dx)) + "x" + str(int(dx)) + ".h5", 'w')

    dset_v_inv = f.create_dataset("vandermonde_inv", (len(node_list),6,6), dtype='f8')
    dset_v_inv[:,:,:] = V_inv[:,:,:]

    dset_cell_ID = f.create_dataset("cell_ID", (int(180/dx),int(360/dx)), dtype='i')
    dset_cell_ID[:,:] = cell_pos[:,:].T

N = int(sys.argv[1])

dx = 2.0 # lat-lon grid spacing in degrees

Grid = ReadGrid.read_grid(N)

# GET NODES LIST
nodes = Grid.nodes

# CREATE LAT-LON GRID
ll_lat = np.arange(90, -90, -dx, dtype=np.float)
ll_lon = np.arange(0, 360, dx, dtype=np.float)

# CONVERT TO RADIANS
ll_lat = np.deg2rad(ll_lat)
ll_lon = np.deg2rad(ll_lon)

# CREATE 2D ARRAY FOR CELL ID'S IN THE LAT-LON GRID
gd2ll_ID = np.zeros((len(ll_lon), len(ll_lat)), dtype=np.int)

# ASSIGN POLAR CELL ID
gd2ll_ID[0, 0] = Grid.npole_node.ID

lat1 = Grid.npole_node.lat
lon1 = Grid.npole_node.lon

f_num = 6
ID_C = gd2ll_ID[0, 0]
ID_MASTER = ID_C

v_max = 0.0
V_inv = np.zeros((len(nodes),6,6))
for y in range(len(ll_lat)):
    for x in range(len(ll_lon)):

        lat2 = ll_lat[y]
        lon2 = ll_lon[x]

        print("FINDING GEODESIC CELL ID FOR POS", np.degrees(lat2), np.degrees(lon2))

        cent_dist = np.ones(6, dtype=np.float)*10.0
        friend_nums = np.ones(6, dtype=np.int)*-1

        fl1 = False
        fl2 = False

        count = 0
        cn = 0
        while (cn%2 == 0):

            lat1 = nodes[ID_MASTER].lat
            lon1 = nodes[ID_MASTER].lon

            p1_x, p1_y = sph2map(lat1, lon1, lat2, lon2)
            p2_x = p1_x
            p2_x += 1.6
            p2_y = p1_y

            # print("SEARCHING NODE ID", ID_C)

            cn = 0
            if nodes[ID_C].friends[-1] < 0:
                f_num = 5
            else:
                f_num = 6

            friend_nums[count] = ID_C
            cent2interp_distance = 10.0
            for i in range(f_num):
                c1_lat, c1_lon = nodes[ID_C].centroids[i%f_num]
                c2_lat, c2_lon = nodes[ID_C].centroids[(i+1)%f_num]

                e1_x, e1_y = sph2map(lat1, lon1, c1_lat, c1_lon)
                e2_x, e2_y = sph2map(lat1, lon1, c2_lat, c2_lon)

                cn += does_intersect(p1_x, p1_y, p2_x, p2_y, e1_x, e1_y, e2_x, e2_y)


            if cn%2 == 1:
                print("POINT LIES INSIDE THE POLYGON", ID_C, ", MOVE ON\n")
                gd2ll_ID[x, y] = ID_C
                ID_MASTER = ID_C

                V_inv[ID_C] = getVandermondeMatrix(nodes, ID_C, inv=True)

                v_max = max(v_max, np.amax(V_inv[ID_C]))
                print("V MAX:", v_max)


            else:
                if not fl1 and not fl2:
                    fl1 = True
                    print("POINT DOES NOT LIE INSIDE THE POLYGON. SEARCHING FRIEND LVL 1.")

                if fl1 and not fl2:
                    ID_C = nodes[ID_MASTER].friends[count]
                    count += 1

                if count >= f_num and fl1 and not fl2:
                    fl1 = False
                    fl2 = True
                    ID_MASTER_OLD = ID_MASTER
                    print("POINT DOES NOT LIE INSIDE ANY FRIENDS. SEARCHING FRIEND LVL 2.")

                if fl2 and not fl1 and count >= f_num:
                    # ID_MASTER = nodes[ID_MASTER_OLD].friends[0]
                    # ID_C = ID_MASTER
                    count = 0
                    count2 = 0

                if fl2 and not fl1:
                    ID_C = nodes[nodes[ID_MASTER].friends[count2]].friends[count]

                    count += 1
                    if count >= f_num:
                        count2 += 1
                        count = 0


# test_interp("/home/hamish/Research/GeodesicODIS/DATA_SPH_1/data.h5",
#             39,
#             Grid,
#             ll_lat,
#             ll_lon,
#             gd2ll_ID,
#             V_inv)

SaveVandermonde2HDF5(N, dx, nodes, V_inv, gd2ll_ID)
