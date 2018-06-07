import matplotlib as mpl
# mpl.use('Agg')
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
    N = int(sys.argv[1])        # geodesic grid recursion level

    dx = float(sys.argv[2])     # lat-lon grid spacing in degrees
    dx_r = np.radians(dx)

    r = 1.0                     # spherical radius

    tolerance = 1.0 - 1e-7

    # Load geodesic grid corresponding to level N
    Grid = ReadGrid.read_grid(N)

    # Get a list of and number of nodes in the geodesic grid
    nodes = Grid.nodes
    gg_node_num = len(nodes)

    # Create lat-lon grid in radians
    ll_lat = np.radians(np.arange(90, -90, -dx, dtype=np.float))
    ll_lon = np.radians(np.arange(0, 360, dx, dtype=np.float))

    if (len(ll_lat)%2 != 0):
        raise ValueError("Lat-lon grid must have an even number of points in latitude, not %d" % len(ll_lat))

    if int(sys.argv[3]):
        test_interp(N, Grid, ll_lat, ll_lon, dx, r)
        sys.exit()

    print(dx, "degree resolution lat-lon grid generated. Max spherical harmonic expansion is", len(ll_lat)/2 - 1)

    ll_node_num = len(ll_lat)*len(ll_lon)   # number of nodes on the lat-lon grid

    ID_MASTER = 0   # master cell ID for geodesic grid
    ll_count = 0    # counter for each lat-lon cell

    # Create 3D array to holding geodesic grid IDs for each lat-lon cell
    mapping_IDS_1D = []
    weights_list_1D = []
    rows_1D = []

    # Handle the north pole cells. Currently they are mapped directly to the
    # pole cell in the geodesic grid. Maybe there is a better way to handle this?
    for x in range(len(ll_lon)):
        mapping_IDS_1D.append(Grid.npole_node.ID)
        weights_list_1D.append([1.0, 0.0, 0.0])
        rows_1D.append(ll_count)
        ll_count += 1


    for y in range(1, len(ll_lat)):
        print("Finding geodesic grid cells intersecting with latitudinal band %3.1f" % np.degrees(ll_lat[y]))
        for x in range(len(ll_lon)):
            polygon_to_plot = []        # List of polygons to plot. Used for debugging purposes.
            does_not_intersect = True   # Switch to indicate when a cell on grid a and b intersect.

            lat1 = ll_lat[y]            # Primary lat-lon of the target grid
            lon1 = ll_lon[x]            # cell. Used by sph2map projection.


            # Define lat-lon cell control volume using the half-way distance
            # between lat-lon grid points.

            p2_1 = sph2map(lat1, lon1, lat1-dx_r*0.5, lon1-dx_r*0.5, r)
            p2_2 = sph2map(lat1, lon1, lat1-dx_r*0.5, lon1+dx_r*0.5, r)
            p2_3 = sph2map(lat1, lon1, lat1+dx_r*0.5, lon1+dx_r*0.5, r)
            p2_4 = sph2map(lat1, lon1, lat1+dx_r*0.5, lon1-dx_r*0.5, r)


            # CREATE POLYGON OUT OF THE LAT-LON GRID POINTS
            # Note that, this could be a more general algorithm if the control
            # volume of each grid was a polygon object in the grid object itself.
            # That way, any shape of polygon can be used in either grid.

            p2=Polygon([(p2_1[0], p2_1[1]),       # This the destination polygon.
                        (p2_2[0], p2_2[1]),       # In the loops below, we attempt
                        (p2_3[0], p2_3[1]),       # to find every cell on the original
                        (p2_4[0], p2_4[1])])      # that intersects with this one.

            polygon_to_plot.append(p2)

            area_ll = p2.area           # Area of the destination grid cell
            area_gg = 0.0               # Total area of the intersections between
                                        # parent and target grid.

            count = 0
            level = 0   # Counter for the level of friends needed to be searched
                        # i.e., immediate friends, level=0, friends of friends,
                        # level=1, etc

            # Array for node IDs that will be searched to find if they intersect
            # with the lat-lon cell.
            node_list = [ID_MASTER]

            searched = []                   # All cell IDs that have been searched
            intersect_list = []             # IDs of cells that intersect with p2
            difference_polygons = []        # Polygons formed from the intersection
                                            # of p1 and p2.
            polygons_that_intersect = []    # Polygons (cells) that intersect with p2

            # Search for intersecting cells (p1) until the total area of every
            # intersection matches the destination cell area (p2), to withtin some
            # tolerance
            while (area_gg < area_ll*tolerance):
                does_not_intersect = True

                # Search over cells in the parent grid (p1) until one is found to
                # intersect with the destination grid (p2).
                while (does_not_intersect):
                    curr_node = Grid.nodes[node_list[count]]

                    f_num = 6
                    if curr_node.friends[-1] < 0: f_num = 5     # hexagonal cell

                    # Loop over all nodes in the control volume to create vertices
                    # of the polygon. Could probably vectorize this step?
                    p1 = []
                    for i in range(f_num):
                        c1_lat, c1_lon = curr_node.centroids[i%f_num]
                        c2_lat, c2_lon = curr_node.centroids[(i+1)%f_num]

                        e1_x, e1_y = sph2map(lat1, lon1, c1_lat, c1_lon, r)
                        e2_x, e2_y = sph2map(lat1, lon1, c2_lat, c2_lon, r)

                        p1.append( (e1_x, e1_y) )

                    # Create the polygon object out of the control volume vertices
                    p1 = Polygon(p1)

                    # Check if the two cells intersect
                    if (p1.intersects(p2)):
                        # Update master node ID
                        ID_MASTER = node_list[count]

                        # Indicate that intersection has occured
                        does_not_intersect = False

                        # Indicate that this node has already been searched
                        searched.append(curr_node.ID)

                        # Store the ID of this node in the intersection list
                        intersect_list.append(curr_node.ID)

                        # Store relevant row in the lat-lon vector
                        rows_1D.append(ll_count)

                        # Store the polygon created by the intersection of p1
                        # and p2
                        difference_polygons.append(p1.intersection(p2))

                        # Add the area of this interesction polygon to the area
                        # sum, and store the intersecting (geodesic) polygon
                        poly_area = difference_polygons[-1].area
                        area_gg += poly_area

                        polygons_that_intersect.append(p1)

                        # Reset the counter, remove search cells from the node_list,
                        # and update the node_list if it is empty
                        count = 0
                        node_list = [node for node in node_list if node not in intersect_list]
                        node_list = [node for node in node_list if node not in searched]
                        node_list = [node for node in node_list if node >= 0]

                        while (len(node_list) == 0):
                            node_list = Grid.get_friends_list(ID_MASTER, level=level)
                            node_list = [node for node in node_list if node not in searched]
                            node_list = [node for node in node_list if node >= 0]
                            level+=1

                    # p1 does not intersect with p2, so we must move onto the next
                    # cell
                    else:
                        # Store the fact that the current cell has been searched to
                        # prevent searching over the same cell twice
                        searched.append(curr_node.ID)
                        count += 1

                        # If we have searched all nodes that exist in the current
                        # list, we must extend to the next set of neighbouring
                        # cells around the MASTER node.
                        if count >= len(node_list)-1:
                            count = 0

                            # Get next level of neighbouring cell IDs
                            node_list = Grid.get_friends_list(ID_MASTER, level=level)

                            # Remove cells that have already been searched
                            node_list = [node for node in node_list if node not in searched]
                            level += 1

            ll_count += 1

            # if ll_lat[y] < np.radians(30.):
            #     for polygon in polygon_to_plot:
            #         xp, yp = polygon.exterior.xy
            #         # print(xp, yp)
            #         plt.plot(xp,yp)
            #     plt.show()

            # All intersecting cells for latlon[x][y] have now been found.
            # Next we assign the starting parent cell ID for the next search,
            # and compute all weighting coefficients for currrent lat-lon cell
            # (p2)

            # Set parent cell ID for next search using the cell that intersects
            # the most with lat-lon cell (p2)
            ID_MASTER = intersect_list[np.argmax(poly_area)]

            # Add all intersecting parent cell ID's to a master list of cell
            # intersections
            mapping_IDS_1D.extend(intersect_list)

            # Get mapping weights from all relvant polygons
            weights_list = get_interpolation_weights(lat1, lon1, p2, difference_polygons, polygons_that_intersect, r)
            weights_list_1D.extend(weights_list)


    # All intersecting cells have been found for every cell in the destination
    # grid. All weighting coefficients have also been calculated. Now we
    # construct a sparse matrix in CSR format, and save that matrix and its
    # column/row indexes to an hdf5 file.

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

    # Create sparse interpolating matrix
    sparse_mapping_matrix = csr_matrix((data, (rows, cols)), shape=(len(ll_lat)*len(ll_lon), 3*len(Grid.nodes)))
    sparse_mapping_matrix.eliminate_zeros()

    indices = sparse_mapping_matrix.indices     # column indexes
    indptr  = sparse_mapping_matrix.indptr      # row indexes
    data    = sparse_mapping_matrix.data        # non-zero data array

    print("Mapping matrix shape:", sparse_mapping_matrix.shape)
    print("Mapping matrix non-zero elements:", len(data))
    print("Mapping matrix sparsity:", len(data)/(sparse_mapping_matrix.shape[0]*sparse_mapping_matrix.shape[1]))

    SaveWeightingMatrix(N, dx, indices, indptr, data)




def get_interpolation_weights(lat1, lon1, polygon_primary, intersection_polygons, gg_polygons, r):
    """
    Function to find the interpolation weights for a cell (primary_polygon), given
    all the cells that intersect it (gg_polygons) and the polygons generated by
    the intersection of primary_polygon and gg_polygons (intersection_polygons).
    """

    area_primary = polygon_primary.area
    intersect_no = len(intersection_polygons)

    weights = [[0., 0., 0.,] for i in range(intersect_no)]

    # The following functions represent the integrands of equations 12 - 14 in
    # Jones (1998). The first integrand is used for 1st order accurate conservative
    # mapping, while the second and third are used for 2nd order accurate
    # conservative mapping.

    def weight1_integrand(lats, lons):
        return -np.sin(lats)

    def weight2_integrand(lats, lons):
        return -np.cos(lats) - lats*np.sin(lats)

    def weight3_integrand(lats, lons):
        return -0.5*lons*(np.sin(lats)*np.cos(lats) + lats)

    # Loop over every polygon that intersects with polygon_primary and evaluate
    # the three weighting coefficients for each one.
    for i in range(intersect_no):
        # Calculate weight one
        integral = line_integral(intersection_polygons[i], weight1_integrand, lat1, lon1, r)
        weights[i][0] = integral/area_primary


        # Calculate weight two
        integral = line_integral(intersection_polygons[i], weight2_integrand, lat1, lon1, r)
        weights[i][1] = integral/area_primary

        integral = line_integral(gg_polygons[i], weight2_integrand, lat1, lon1, r)
        weights[i][1] -= weights[i][0]/gg_polygons[i].area * integral


        # Calculate weight three
        integral = line_integral(intersection_polygons[i], weight3_integrand, lat1, lon1, r)
        weights[i][2] = integral/area_primary

        integral = line_integral(gg_polygons[i], weight3_integrand, lat1, lon1, r)
        weights[i][2] -= weights[i][0]/gg_polygons[i].area * integral

    return weights




def line_integral(polygon, integrand_function, lat1, lon1, r):
    """
    Function to perfrom a line integral around a the edge of polygon, using
    some perscribed function of latitude and longitude: integrand_function.
    """

    xp, yp = polygon.exterior.xy
    integral = 0.0
    n = 150
    for j in range(len(xp)-1):
        xs = np.linspace(xp[j], xp[j+1], n)
        ys = np.linspace(yp[j], yp[j+1], n)

        lats, lons, sinLats = map2sph(lat1, lon1, xs, ys, r, trig=True)

        lons_norm = lons - lon1

        integrand = integrand_function(lats, lons_norm)

        integral += -np.trapz(integrand, lons)

    return integral




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




def test_interp(N, grid, ll_lat, ll_lon, dx, r):
    """
    Test the interpolation of geodesic grid N to latlon resolution dx. Returns
    the max and mean error of the interpolation.
    """

    file_name = "grid_l" + str(N) + "_" + str(int(dx)) + "x" + str(int(dx)) + "_weights.h5"

    f= h5py.File(file_name, 'r')

    cols = f["column index"]
    rows = f["row index"]
    w    = f["weights"]

    map_matrix = csr_matrix((w, cols, rows), shape=(len(ll_lat)*len(ll_lon), 3*len(grid.nodes)))

    m = 2.
    n = 1.

    ll_x, ll_y = np.meshgrid(ll_lon, ll_lat)

    tf_ll = np.cos(m*ll_x) * np.cos(n*ll_y)**4.

    gg_lat, gg_lon = np.array(grid.lats), np.array(grid.lons)
    triang = tri.Triangulation(gg_lon, gg_lat)

    tf_gg = np.cos(m*gg_lon) * np.cos(n*gg_lat)**4.
    tf_gg_dlat = 1./r * (-4.*n*np.sin(n*gg_lat)*np.cos(n*gg_lat)**3. *np.cos(m*gg_lon))
    tf_gg_dlon = 1./r * (-m*np.sin(m*gg_lon)*np.cos(n*gg_lat)**4.)/np.cos(gg_lat)

    gg_data = np.zeros(3*len(grid.nodes))
    ll_interp = np.zeros(len(ll_lat)*len(ll_lon))

    for i in range(len(grid.nodes)):
        gg_data[3*i]        = tf_gg[i]
        gg_data[3*i + 1]    = tf_gg_dlat[i]
        gg_data[3*i + 2]    = tf_gg_dlon[i]

    ll_interp = map_matrix.dot(gg_data)
    ll_interp = np.reshape(ll_interp, (len(ll_lat), len(ll_lon)))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16,3.5))

    vmax = max(np.amax(ll_interp), max(np.amax(tf_ll), np.amax(tf_gg)))
    vmin = min(np.amin(ll_interp), min(np.amin(tf_ll), np.amin(tf_gg)))

    levels = np.linspace(vmin, vmax, 9)
    levels2 = 1e3*np.linspace(np.amin(ll_interp-tf_ll), np.amax(ll_interp-tf_ll), 9)

    axes = [ax1, ax2, ax3, ax4]

    c1 = ax1.contourf(ll_lon, ll_lat, tf_ll, levels=levels)
    c2 = ax2.tricontourf(triang, tf_gg, levels=levels)
    c3 = ax3.contourf(ll_lon, ll_lat, ll_interp, levels=levels)
    c4 = ax4.contourf(ll_lon, ll_lat, 1e3*(ll_interp-tf_ll))

    c = [c1, c2, c3, c4]

    for cb, ax in zip(c,axes):
        ax.set_ylim([-np.pi*0.5, np.pi*0.5])
        ax.set_aspect('equal')
        plt.colorbar(cb, ax=ax, orientation='horizontal')

    max_err = np.amax(abs(ll_interp-tf_ll))
    mean_err = np.mean(abs(ll_interp.flatten()-tf_ll.flatten()))

    print("Max error:", max_err, ", Mean error:", mean_err)

    ax1.set_title("Lat-lon grid (analytic)")
    ax2.set_title("Geodesic grid (analytic)")
    ax3.set_title("Interpolated solution")
    ax4.set_title("Interpolated - analytic solution ($\\times 10^3$)")

    # fig.savefig("/home/hamish/Dropbox/conservative_interp_test_g6_1x1.pdf")
    plt.show()

    return max_err, mean_err




def SaveWeightingMatrix(N, dx, cols, rows, data):
    """
    Save the weighting coefficients (data) to an hdf5 file. Cols and rows are
    the row and column indices, in csr format, and are also saved.
    """

    file_name = "grid_l" + str(N) + "_" + str(int(dx)) + "x" + str(int(dx)) + "_weights.h5"
    print("Saving mapping matrix in CSR format to " + file_name)

    f = h5py.File(file_name, 'w')

    dset_cols = f.create_dataset("column index", (len(cols), ), dtype='i')
    dset_cols[:] = cols[:]

    dset_rows = f.create_dataset("row index", (len(rows), ), dtype='i')
    dset_rows[:] = rows[:]

    dset_data = f.create_dataset("weights", (len(data), ), dtype='d')
    dset_data[:] = data[:]

    dset_data.attrs["non-zero elements"] = len(data)

    f.close()


if __name__=='__main__':
    main()
