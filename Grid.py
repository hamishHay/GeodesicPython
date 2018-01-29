import numpy as np
from Node import Node

class Grid:
    def __init__(self):
        self.node_num = 0
        self.node_list = []
        self.friends_list = []


    def add_node(self, node):
        self.node_list.append( node )


    def find_friends(self):
        for i in range(len(self.node_list)):
            friends = []
            for j in range(len(self.node_list)):
                if i != j:
                    A = self.node_list[i]
                    B = self.node_list[j]

                    dist = np.sqrt((np.sum((A-B)**2)))

                    if dist < 1.6:
                        self.node_list[i].add_friend(self.node_list[j])
                        friends.append(j)
            friends.append(-1)
            self.friends_list.append(friends)

    def bisect_grid(self):
        print("\t Bisecting edges...")

        new_nodes = []
        node_count = len(self.node_list)-1
        min_dist = 1.0
        # Step 1: Bisect edges and update the inter_nodes
        for i in range(len(self.node_list)):
            # set parent node
            node = self.node_list[i]

            # loop through friends, finding the edge bisect
            for j in range(node.f_num):
                if not node.updated[j]:
                    # set friend node
                    friend = self.node_list[i].friends[j]

                    # calculate midpoint between node and friend
                    middle = 0.5*(node + friend)
                    middle = middle/np.sqrt(np.sum((middle**2)))

                    # create new node based on the midpoint location
                    node_count += 1
                    new_node = Node(middle[0], middle[1], middle[2], node_count)

                    # project new node onto sphere
                    new_node_xyz = new_node.coords_cart
                    mag = np.sqrt(np.sum((new_node_xyz**2)))
                    new_node_xyz = new_node_xyz/mag

                    min_dist = min(min_dist, np.sqrt(np.sum((node - new_node)**2)))

                    # add the new node to the intermediate friend list
                    node.inter_friends[j] = new_node

                    new_nodes.append(new_node)

                    new_node.friends.append(node)
                    new_node.friends.append(friend)

                    # update the intermediate friend list for the current friend node
                    for k in range(friend.f_num):
                        # set friend of a friend
                        friend2 = friend.friends[k]

                        # the current node, and friend of the friend node are the same
                        # then they must share the new_node
                        if node == friend2:
                            friend.inter_friends[k] = new_node
                            friend.updated[k] = 1
                            break

                    # set the update list to inform the node that this friend
                    # has been updated
                    node.updated[j] = 1

                    # print()


        # Step 2: Add the other friends for newly created nodes
        for i in range(len(self.node_list)):
            # set parent node
            node = self.node_list[i]

            # Loop through intermediate friends to add the extra two friends
            added = 0
            for j in range(node.f_num):
                # set intermediate node
                inter_friend = node.inter_friends[j]

                for k in range(node.f_num):
                    # set next intermediate node
                    inter_friend2 = node.inter_friends[k]

                    # if the intermediate nodes aren't equivalent, find the distance
                    # between them
                    if inter_friend != inter_friend2:
                        dist = np.sqrt(np.sum((inter_friend - inter_friend2)**2))

                        # if the distance is less than the min, add one intermediate
                        # node to the others friend list
                        if dist < min_dist*1.2:
                            inter_friend.friends.append(inter_friend2)

                    if added == 2:
                        break

        # Step 3: Move inter_friends into friend list and reset
        for i in range(len(self.node_list)):
            # set parent node
            node = self.node_list[i]

            for j in range(node.f_num):
                node.friends[j] = node.inter_friends[j]
                node.inter_friends[j] = 0
                node.updated[j] = 0

        # Step 4: add all new nodes to master node list
        self.node_list.extend(new_nodes)

        # count = 0
        # for i in range(len(self.node_list)):
        #     node1 = self.node_list[i]
        #     point1 = np.degrees(node1.coords_sph[1:])
        #     if abs(point1[0]) < 0.01:
        #         count += 1
        #
        # print("COUNT", count)

    def create_master_friends_list(self):
        print("\t Compiling master friends list...")
        self.friends_list = np.zeros((len(self.node_list), 6), dtype=np.int)

        for i in range(len(self.node_list)):
            node = self.node_list[i]

            for j in range(node.f_num):
                friend = node.friends[j]
                self.friends_list[i][j] = int(friend.ID) #self.node_list.index(friend)

            if node.f_num == 5:
                self.friends_list[i][-1] = -1

    def order_friends(self):
        print("\t Ordering friends in clockwise fashion...")
        def length(v):
            return np.sqrt(v[0]**2+v[1]**2)
        def dot_product(v,w):
           return v[0]*w[0]+v[1]*w[1]
        def determinant(v,w):
           return v[0]*w[1]-v[1]*w[0]
        def inner_angle(v,w):
           cosx=dot_product(v,w)/(length(v)*length(w))
           rad=np.arccos(cosx) # in radians
           return rad*180/np.pi # returns degrees
        def angle_clockwise(A, B):
            inner=inner_angle(A,B)
            det = determinant(A,B)
            if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
                return inner
            else: # if the det > 0 then A is immediately clockwise of B
                return 360-inner

        for i in range(len(self.node_list)):
            #Convert central point to spherical coords
            node1 = self.node_list[i]
            point1 = node1.coords_sph[1:]

            lat1 = point1[0]
            lon1 = point1[1]

            p0 = [0,1]

            friend_coords = []
            angles = []

            new_friends = []

            friend_num = node1.f_num

            pentagon = False
            for j in range(6):
                f_index = self.friends_list[i][j]

                if f_index != -1:
                    node2 = self.node_list[f_index]
                    coords = node2.coords_sph[1:]

                    lat2 = coords[0]
                    lon2 = coords[1]

                    friend_coords.append(self.map_coords(lat1,lat2,lon1,lon2))

                    p1 = friend_coords[-1]

                    angles.append(angle_clockwise(p0, p1))
                else:
                    pentagon = True


            friends = []
            for j in np.argsort(angles):
                friends.append(int(self.friends_list[i][j]))

            if pentagon:
                friends.append(-1)

            self.friends_list[i] = np.array(friends)

    def map_coords(self,lat1,lat2,lon1,lon2):
        m = 2.0 * (1.0 + np.sin(lat2)*np.sin(lat1) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
        x = m * np.cos(lat2) * np.sin(lon2 - lon1)
        y = m * (np.sin(lat2)*np.cos(lat1) - np.cos(lat2)*np.sin(lat1)*np.cos(lon2-lon1))

        return [x, y]

    def find_arc_midpoint(self, lat1, lat2, lon1, lon2):
        dlon = lon2-lon1

        Bx = np.cos(lat2)*np.cos(dlon)
        By = np.cos(lat2)*np.sin(dlon)
        #
        latM = np.arctan2(np.sin(lat1)+np.sin(lat2), \
                          np.sqrt((np.cos(lat1) + Bx)**2. + By**2.))
        lonM = lon1 + np.arctan2(By, np.cos(lat1) + Bx)

        # latM = np.arctan2(np.sqrt((np.cos(lat1) + Bx)**2. + By**2.), np.sin(lat1)+np.sin(lat2))
        # lonM = lon1 + np.arctan2(np.cos(lat1) + Bx, By)

        return latM, lonM

    def find_centers(self):
        # from scipy.spatial import SphericalVoronoi
        print("\t Locating element centres...")
        self.centers = np.zeros((len(self.node_list),6,3))
        self.centroids = np.zeros((len(self.node_list),6,2))
        for i in range(len(self.node_list)):
            p0 = self.node_list[i].coords_cart

            total = self.node_list[i].f_num
            centers_ = np.ones((6,3))*-1
            centers_sph_ = np.ones((6,2))*-1

            for j in range(total):
                p1 = self.node_list[int(self.friends_list[i][j])].coords_cart
                p2 = self.node_list[int(self.friends_list[i][(j+1)%total])].coords_cart

                center = (p0 + p1 + p2)/3.0

                mag = np.sqrt(sum(center**2))

                center = center/mag

                centers_[j] = center

                centers_sph_[j] = self.cart2sph(center)[1:]

                # test_c = np.cross((p2 - p0),(p1 - p0))
                # test_c = test_c/np.sqrt(sum(test_c**2))

                # centers_sph_[j] = self.cart2sph(test_c)[1:]

                # print(test_

                # vor = SphericalVoronoi(np.array([p0,p1,p2]))
                # print(centers_sph_[j], self.cart2sph(test_c)[1:])

            self.centers[i] = centers_
            self.centroids[i] = centers_sph_

    def shift_centroids(self):
        new_centroids_sph = np.ones((len(self.node_list), 3))
        new_centroids_cart = np.ones((len(self.node_list), 3))
        for i in range(len(self.node_list)):

            # DETERMINE IF HEXAGON OR PENTAGON
            f_num = 6
            if self.friends_list[i][-1] == -1:
                f_num = 5

            c1 = self.node_list[i].coords_sph[1:]
            c1_c = self.node_list[i].coords_cart
            sub_areas = np.zeros(f_num)
            sub_centroids = np.zeros((f_num, 3))
            for j in range(f_num):
                c2 = np.radians(self.centroids[i, j%f_num])
                c3 = np.radians(self.centroids[i, (j+1)%f_num])

                sub_area = self.sphericalArea(c1,c2,c3)
                sub_areas[j] = sub_area

                c2_c = self.centers[i, j%f_num]
                c3_c = self.centers[i, (j+1)%f_num]

                cx, cy, cz = zip(c1_c, c2_c, c3_c)
                sub_centroids[j] = np.array([sum(cx)/3.0,
                                             sum(cy)/3.0,
                                             sum(cz)/3.0])

            total_area = np.sum(sub_areas)

            centroid_3d_cart = np.zeros(3)
            for j in range(f_num):
                for k in range(3):
                    centroid_3d_cart[k] += sub_centroids[j,k]*sub_areas[j]

            mag = np.sqrt(sum(centroid_3d_cart**2))
            centroid_3d_cart /= mag

            new_centroids_cart[i] = centroid_3d_cart
            new_centroids_sph[i] = self.cart2sph(centroid_3d_cart)
            # print(np.degrees(c1), new_centroids[i])

        for i in range(len(self.node_list)):
            self.node_list[i].coords_sph = np.radians(new_centroids_sph[i])
            self.node_list[i].coords_cart = new_centroids_cart[i]

    def spring_dynamics(self,N):
        M = 100.0
        alpha = 0.5
        dt = 0.1
        beta = 1.19
        k = 0.001
        dbar = beta * 2*np.pi/(10*2**(N-2))


        w_new = np.zeros((len(self.node_list), 3))
        w_old = np.zeros((len(self.node_list), 3))
        r_new = np.zeros((len(self.node_list), 3))
        r_old = np.zeros((len(self.node_list), 3))
        spring_force = np.zeros((len(self.node_list), 3))
        err = np.zeros((len(self.node_list), 3))
        conv_total = 0.0
        conv_total2 = 0.0
        while conv_total2 < 1000.0:
            avg_err = 0
            spring_force *= 0.0
            for i in range(12, len(self.node_list)):
                # DETERMINE IF HEXAGON OR PENTAGON
                f_num = 6
                if self.friends_list[i][-1] == -1:
                    f_num = 5

                # spring_force = np.zeros(3)
                p0 = self.node_list[i].coords_cart
                c0 = self.node_list[i].coords_sph
                r_old[i] = p0
                for j in range(f_num):
                    p1 = self.node_list[int(self.friends_list[i][j])].coords_cart
                    c1 = self.node_list[int(self.friends_list[i][j])].coords_sph

                    e = p1 - p0
                    e /= np.sqrt(sum(e**2.0))

                    d = self.sphericalLength(c0, c1)
                    # d = p1 - p0
                    # d = np.sqrt(sum(d**2))

                    # print(d, d2)
                    # if (d < 0): print("DAMN")

                    spring_force[i] += k*(d - dbar)*e

            gamma = dbar/beta
            avg_err = np.sqrt(np.sum((spring_force/k)**2.0, axis=1))/gamma

            epsilon = 1e-4

            conv = np.ones(len(self.node_list))
            conv[avg_err >= epsilon] = 0.0
            conv_total = sum(conv)/len(self.node_list) * 100.0
            if conv_total > 99.999:
                conv_total2 += 1.0
            # print(conv_total)

            # avg_err += err
            # if (i == 20):
            #     print(np.sqrt(sum(spring_force**2.0))/gamma)
            #
            #     print(dbar/beta)

            spring_force -= alpha*w_old
            spring_force /= M

            w_new = dt*spring_force + w_old
            w_old = w_new

            r_new = dt*w_new + r_old
            r_new /= np.sqrt(sum(r_new**2.0))
            # print(r_new, r_old)
            T = 20
            print(conv_total, avg_err[T], spring_force[T], w_new[T], self.cart2sph(r_new[T])[1:], np.sqrt(sum((r_new[T]-r_old[T])**2.0)))

            # w_old = (r_new - r_old)/dt
            r_old = r_new
            #
            # print(avg_err/len(self.node_list))

            for i in range(12, len(self.node_list)):
                self.node_list[i].update_xyz(r_new[i])

        for i in range(12, len(self.node_list)):
            r_new /= np.sqrt(sum(r_new**2.0))
            self.node_list[i].update_xyz(r_new[i])


    def sphericalArea(self, c1, c2, c3):
        a = self.sphericalLength(c1,c2)
        b = self.sphericalLength(c2,c3)
        c = self.sphericalLength(c3,c1)

        A = np.arccos((np.cos(a) - np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)))
        B = np.arcsin(np.sin(A)*np.sin(b)/np.sin(a))
        C = np.arcsin(np.sin(A)*np.sin(c)/np.sin(a))

        E = (A + B + C) - np.pi

        return abs(E)
        # self.sphericalLength()

    def sphericalLength(self, c1, c2):
        if len(c1) == 3:
            i = 1
        else:
            i = 0
        lat1 = c1[i]
        lat2 = c2[i]
        dlat = lat2 - lat1
        dlon = c2[i+1] - c1[i+1]

        # ang = np.sin(lat1)*np.sin(lat2)
        # ang += np.cos(lat1)*np.cos(lat2)*np.cos(dlon)
        # ang = np.arccos(ang)

        ang = np.sin(dlat*0.5)**2.0
        ang += np.cos(lat1)*np.cos(lat2)*np.sin(dlon*0.5)**2.0
        ang = 2.*np.arcsin(np.sqrt(ang))

        return ang

    def sph2cart(self,r,theta,phi):
        theta = theta
        phi = phi

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return np.array([x, y, z])


    def cart2sph(self, coords):
        x = coords[0]
        y = coords[1]
        z = coords[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.pi*0.5 - np.arccos(z/r)
        lon = np.arctan2(y,x)


        if np.rad2deg(lon)+180.0 > 359.9:
            return np.array([r, np.rad2deg(lat), 0.0])
        return np.array([r, np.rad2deg(lat), np.rad2deg(lon)+180.0])

    # def sph2cart(self,coords):


    def save_grid(self, filename):
        print("\t Saving grid to " + filename)

        lons = []
        lats = []
        centers = np.ones((len(self.node_list), 6))*-1
        for i in range(len(self.node_list)):
            coords = np.degrees(self.node_list[i].coords_sph[1:])
            lons.append(coords[1])
            lats.append(coords[0])

        f = open(filename,'w')

        f.write('{:<5s} {: <10s}   {: <10s}   {: <36s}  {:20s}'.format("ID", "NODE_LAT", "NODE_LON", "FRIENDS LIST", "CENTROID COORD LIST\n"))

        for i in range(len(self.node_list)):
            node = self.node_list[i]
            f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
            string = '{'
            for j in range(len(self.friends_list[0])):
               string += '{:4d}'.format(int(self.friends_list[i][j]))
               if j < 5:
                   string += ', '
               else:
                   string += '}, '

            string += '{'
            for j in range(len(self.centers[0])):
                lat_c, lon_c = self.cart2sph([self.centers[i][j][0], self.centers[i][j][1], self.centers[i][j][2]])[1:]
                if node.f_num == 5 and j == node.f_num:
                    lat_c = -1.0
                    lon_c = -1.0
                string += '({:10.6f}, {:10.6f})'.format(lat_c, lon_c)
                if j < len(self.centers[0]) - 1:
                    string += ', '
                else:
                    string += '} \n'

            f.write(string)
        f.close()
