import ReadGrid
import numpy as np
from numpy import deg2rad
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon

def main():
    N = int(sys.argv[1])
    dx = float(sys.argv[2]) # lat-lon grid spacing in degrees

    r = 252.1e3

    dx_r = np.radians(dx)
    tolerance = 1.0 - 1e-7

    Grid = ReadGrid.read_grid(N)

    # GET NODES LIST
    nodes = Grid.nodes
    gg_node_num = len(nodes)

    # CREATE LAT-LON GRID IN RADIANS
    ll_lat = np.radians(np.arange(90, -90, -dx, dtype=np.float))
    ll_lon = np.radians(np.arange(0, 360, dx, dtype=np.float))

    if (len(ll_lat)%2 != 0):
        raise ValueError("Lat-lon grid must have an even number of points in latitude, not %d" % len(ll_lat))

    print(len(ll_lat)/2 - 1)

    ll_node_num = len(ll_lat)*len(ll_lon)

    # CREATE 2D ARRAY FOR CELL ID'S IN THE LAT-LON GRID
    gd2ll_ID = np.zeros((len(ll_lon), len(ll_lat)), dtype=np.int)

    # ASSIGN POLAR CELL ID
    gd2ll_ID[0, 0] = Grid.npole_node.ID

    ID_C = gd2ll_ID[0, 0]
    ID_MASTER = ID_C

    # Create 3D array to holding geodesic grid IDs for each lat-lon cell
    mapping_IDS = [[[] for j in range(len(ll_lon))] for i in range(len(ll_lat))]
    weights_list = [[[] for j in range(len(ll_lon))] for i in range(len(ll_lat))]
    mapping_IDS_1D = []
    weights_list_1D = []
    rows_1D = []

    ll_count = 0    # counter for each lat-lon cell
    for x in range(len(ll_lon)):
        mapping_IDS[0][x] = [Grid.npole_node.ID]
        mapping_IDS_1D.append(Grid.npole_node.ID)
        weights_list_1D.append([1.0, 0.0, 0.0])
        rows_1D.append(ll_count)
        ll_count += 1



    for y in range(1, len(ll_lat)):
        print("Finding geodesic grid cells intersecting with latitudinal band %3.1f" % np.degrees(ll_lat[y]))
        for x in range(len(ll_lon)):
            polygon_to_plot = []
            does_not_intersect = True

            lat1 = ll_lat[y]
            lon1 = ll_lon[x]

            p2_1 = sph2map(lat1, lon1, lat1-dx_r*0.5, lon1-dx_r*0.5, r)
            p2_2 = sph2map(lat1, lon1, lat1-dx_r*0.5, lon1+dx_r*0.5, r)
            p2_3 = sph2map(lat1, lon1, lat1+dx_r*0.5, lon1+dx_r*0.5, r)
            p2_4 = sph2map(lat1, lon1, lat1+dx_r*0.5, lon1-dx_r*0.5, r)

            # CREATE POLYGON OUT OF THE LAT-LON GRID POINTS
            p2=Polygon([(p2_1[0], p2_1[1]),
            (p2_2[0], p2_2[1]),
            (p2_3[0], p2_3[1]),
            (p2_4[0], p2_4[1])])

            polygon_to_plot.append(p2)

            cent_dist = np.ones(6, dtype=np.float)*10.0
            friend_nums = np.ones(6, dtype=np.int)*-1

            area_ll = p2.area
            area_gg = 0.0

            count = 0
            level = 0
            node_list = [ID_MASTER]

            searched = []
            intersect_list = []
            intersect_areas = []
            intersect_polygons = []
            intersecting_polygons = []

            while (area_gg < area_ll*tolerance):
                does_not_intersect = True
                while (does_not_intersect):
                    curr_node = Grid.nodes[node_list[count]]

                    # print("SEARCHING NODE ", curr_node.ID)

                    f_num = 6
                    if curr_node.friends[-1] < 0: f_num = 5

                    p1 = []
                    for i in range(f_num):
                        c1_lat, c1_lon = curr_node.centroids[i%f_num]
                        c2_lat, c2_lon = curr_node.centroids[(i+1)%f_num]

                        e1_x, e1_y = sph2map(lat1, lon1, c1_lat, c1_lon, r)
                        e2_x, e2_y = sph2map(lat1, lon1, c2_lat, c2_lon, r)

                        p1.append( (e1_x,e1_y) )

                    # CREATE POLYGON OUT OF GEODESIC GRID CONTROL VOLUME
                    p1 = Polygon(p1)

                    if (p1.intersects(p2)):
                        # print("POINT LIES INSIDE THE POLYGON", node_list[count], ", MOVE ON")
                        gd2ll_ID[x, y] = node_list[count]
                        ID_MASTER = node_list[count]

                        # print(np.degrees(lat2-dx_r*0.5), np.degrees(lon2-dx_r*0.5), np.degrees(map2sph(lat1, lon1, p2_1[0], p2_1[1], r)))
                        does_not_intersect = False

                        searched.append(curr_node.ID)
                        intersect_list.append(curr_node.ID)

                        rows_1D.append(ll_count)

                        try:
                            intersect_polygons.append(p1.intersection(p2))
                            poly_area = intersect_polygons[-1].area
                            area_gg += poly_area
                            intersect_areas.append(poly_area)
                        except:
                            print(p2.is_valid)
                            sys.exit()

                        polygon_to_plot.append(p1)
                        intersecting_polygons.append(p1)


                        count = 0
                        node_list = [node for node in node_list if node not in intersect_list]

                        if len(node_list) == 0:
                            node_list = Grid.get_friends_list(ID_MASTER, level=0)
                            node_list = [node for node in node_list if node not in searched]


                    else:
                        searched.append(curr_node.ID)
                        count += 1

                        if count >= len(node_list):
                            # print("POINT DOES NOT LIE INSIDE THE POLYGON. INCREASING FRIEND LEVEL.")

                            count = 0
                            node_list = Grid.get_friends_list(ID_MASTER, level=level)
                            node_list = [node for node in node_list if node not in searched]
                            level += 1
            ll_count += 1

            # print(polygon_to_plot)
            # if ll_lat[y] < np.radians(30.):
            #     for polygon in polygon_to_plot:
            #         xp, yp = polygon.exterior.xy
            #         # print(xp, yp)
            #         plt.plot(xp,yp)
            #     plt.show()

            # Set GG ID for next search using GG cell with most area intersecting
            # with last lat-lon cell
            ID_MASTER = intersect_list[np.argmax(poly_area)]

            # add all intersecting geodesic grid cell ID's to current lat-lon cell
            mapping_IDS[y][x] = intersect_list
            mapping_IDS_1D.extend(intersect_list)

            # get mapping weights from polygon list
            weights_list[y][x] = get_interpolation_weights(lat1, lon1, p2, intersect_polygons, intersecting_polygons, r)
            weights_list_1D.extend(weights_list[y][x])

    # Create 1d arrays for the column, row indexes, and data to convert
    # weightings to csr matrix format
    cols = []
    rows = []
    data = []
    for i in range(len(weights_list_1D)):
        cols.append(3*mapping_IDS_1D[i]    )
        cols.append(3*mapping_IDS_1D[i] + 1)
        cols.append(3*mapping_IDS_1D[i] + 2)

        rows.append(rows_1D[i])
        rows.append(rows_1D[i])
        rows.append(rows_1D[i])

        data.append(weights_list_1D[i][0])
        data.append(weights_list_1D[i][1])
        data.append(weights_list_1D[i][2])

    cols = np.array(cols, dtype=np.int)
    rows = np.array(rows, dtype=np.int)
    data = np.array(data, dtype=np.double)

    # Create dense matrix
    count = 0
    dense_mapping_matrix = np.zeros((ll_node_num, 3*gg_node_num), dtype=np.double)
    for row, col in zip(rows, cols):
            dense_mapping_matrix[row][col] = data[count]
            count+=1

    # Convert dense matrix to spare matrix
    sparse_mapping_matrix = csr_matrix(dense_mapping_matrix)

    indices = sparse_mapping_matrix.indices     # column indexes
    indptr  = sparse_mapping_matrix.indptr      # row indexes
    data    = sparse_mapping_matrix.data        # non-zero data array

    print(sparse_mapping_matrix, sparse_mapping_matrix.shape)
    print(indices, indptr, data, len(data))

    SaveWeightingMatrix(N, dx, indices, indptr, data)


def get_interpolation_weights(lat1, lon1, polygon_primary, intersecting_polygons, gg_polygons, r):
    area_primary = polygon_primary.area
    intersect_no = len(intersecting_polygons)

    weights = [[0., 0., 0.,] for i in range(intersect_no)]

    weight_sum = 0.0
    for i in range(intersect_no):
        # Calculate weight one ###############################################
        ######################################################################

        # Weighting 1 is just the area ratio of the intersecting polygon and
        # destination lat-lon (primary) polgon
        weights[i][0] = intersecting_polygons[i].area/area_primary

        # Calculate weight two ###############################################
        ######################################################################

        n = 50
        xp, yp = intersecting_polygons[i].exterior.xy
        integral = 0.0
        for j in range(len(xp)-1):
            xs = np.linspace(xp[j], xp[j+1], n)
            ys = np.linspace(yp[j], yp[j+1], n)

            lats, lons, sinLats = map2sph(lat1, lon1, xs, ys, r, trig=True)

            func = (-np.cos(lats) - lats*sinLats)

            integral += -np.trapz(func, lons)

        weights[i][1] = integral/area_primary * r**2.0

        xp, yp = gg_polygons[i].exterior.xy
        integral = 0.0
        weight_sum = 0.0
        for j in range(len(xp)-1):
            xs = np.linspace(xp[j], xp[j+1], n)
            ys = np.linspace(yp[j], yp[j+1], n)

            lats, lons, sinLats = map2sph(lat1, lon1, xs, ys, r, trig=True)

            func = (-np.cos(lats) - lats*sinLats)

            integral += -np.trapz(func, lons)

        weights[i][1] -= weights[i][0]/gg_polygons[i].area * integral * r**2.0

        # Calculate weight three #############################################
        ######################################################################

        xp, yp = intersecting_polygons[i].exterior.xy
        integral = 0.0
        for j in range(len(xp)-1):
            xs = np.linspace(xp[j], xp[j+1], n)
            ys = np.linspace(yp[j], yp[j+1], n)

            lats, lons, sinLats = map2sph(lat1, lon1, xs, ys, r, trig=True)

            func = -0.5*lons*(sinLats*np.cos(lats) + lats)

            integral += -np.trapz(func, lons)

        weights[i][2] = integral/area_primary * r**2.0

        xp, yp = gg_polygons[i].exterior.xy
        integral = 0.0
        for j in range(len(xp)-1):
            xs = np.linspace(xp[j], xp[j+1], n)
            ys = np.linspace(yp[j], yp[j+1], n)

            lats, lons, sinLats = map2sph(lat1, lon1, xs, ys, r, trig=True)

            func = -0.5*lons*(sinLats*np.cos(lats) + lats)

            integral += -np.trapz(func, lons)

        weights[i][2] -= weights[i][0]/gg_polygons[i].area * integral * r**2.0

    return weights

def sph2map(lat1,lon1,lat2,lon2, r):
    m = 2.0 / (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    x = m * r * np.cos(lat2) * np.sin(lon2 - lon1)
    y = m * r * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

    if abs(x) < 1e-6: x = 0.0
    if abs(y) < 1e-6: y = 0.0

    return np.array([x, y])

def map2sph(lat1, lon1, x, y, r, trig=False):
    rho = np.sqrt(x**2. + y**2.)
    c = 2.*np.arctan(rho/(2.*r))

    sinLat = np.cos(c)*np.sin(lat1) + y*np.sin(c)*np.cos(lat1)/rho
    lat = np.arcsin(sinLat)

    lon = lon1 + np.arctan(x*np.sin(c)/(rho*np.cos(lat1)*np.cos(c) - y*np.sin(lat1)*np.sin(c)))
    if not trig:
        return np.array([lat, lon])
    else:
        return np.array([lat, lon, sinLat])

def distanceBetween(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def test_interp():
    plt.show()

def SaveWeightingMatrix(N, dx, cols, rows, data):
    f = h5py.File("grid_l" + str(N) + "_" + str(int(dx)) + "x" + str(int(dx)) + "_weights.h5", 'w')

    dset_cols = f.create_dataset("column index", (len(cols), ), dtype='i')
    dset_cols[:] = cols[:]



    dset_rows = f.create_dataset("row index", (len(rows), ), dtype='i')
    dset_rows[:] = rows[:]

    dset_data = f.create_dataset("weights", (len(data), ), dtype='d')
    dset_data[:] = data[:]

    dset_data.attrs["non-zero elements"] = len(data)

    f.close()


if __name__=='__main__':
    try:
        test = int(sys.argv[3])
        test = True
    except IndexError:
        test = False

    if test:
        test_interp()
    else:
        main()

# test_interp("/home/hamish/Research/GeodesicODISBeta/GeodesicODIS/DATA/data.h5",
#             99,
#             Grid,
#             ll_lat,
#             ll_lon,
#             gd2ll_ID,
#             V_inv,
#             Rotation,
#             r)

# SaveVandermonde2HDF5(N, dx, nodes, V_inv, Rotation, gd2ll_ID)
