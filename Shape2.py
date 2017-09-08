
import numpy as np
from numpy import sin, cos, radians, arctan2, sqrt,arccos, tan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from scipy.stats import mode
import sys

class Shape:
    def __init__(self):
        self.vertex_list = []
        self.friends = []
        self.vertexs = []
        self.factor = 1.2
        self.min_mag = 2.0

    def add_vertex(self, vertex):
        self.vertex_list.append(vertex)

    def bisect_edges(self):
        #print("OLD VERTEX LIST", self.vertex_list)
        vertii = [i for i in self.vertex_list]
        for i in range(len(self.vertex_list)):
            p1 = self.vertex_list[i]
            for j in range(6):
                if self.friends[i][j] >= 0:
                    p2 = self.vertex_list[int(self.friends[i][j])]
                    # if ( p1[0] != p2[0] ) and ( p1[1] != p2[1] ) and ( p1[2] != p2[2] ):
                    vector = []
                    for k in range(3):
                        vector.append((p2[k]+p1[k])*0.5)
                    vector = np.array(vector)

                    vertii.append(vector)

        self.vertex_list = np.array(vertii)

        ncols = self.vertex_list.shape[1]

        dtype = self.vertex_list.dtype.descr * ncols

        struct = self.vertex_list.view(dtype)

        uniq = np.unique(struct)

        self.vertex_list = uniq.view(self.vertex_list.dtype).reshape(-1,ncols)





    def get_min_mag(self):
        mags = []
        p1 = self.vertex_list[0]
        for i in range(len(self.vertex_list[1:])):
            p2 = self.vertex_list[i]

            if (p1[0] != p2[0]) and (p1[0] != p2[0]) and (p1[0] != p2[0]):
                mags.append(np.sqrt(sum((p1-p2)**2)))
                # mags[-1] = np.sqrt(mags[-1])
        self.min_mag = min(mags)
        print("MAX: ", self.min_mag)

    def find_friends(self,level=1.2):
        self.get_min_mag()
        paired = []
        counti = np.zeros(len(self.vertex_list))
        self.friends = np.ones((len(self.vertex_list), 6))*-1
        for i in range(len(self.vertex_list)):
            p1 = self.vertex_list[i]
            #if i%100 == 0:
            #    print(float(i)/float(len(self.vertex_list))*100)
            for j in range(len(self.vertex_list)):
                p2 = self.vertex_list[j]
                mag = np.sqrt(sum((p1-p2)**2.0))
                if (mag <= self.min_mag*level and mag > 0.0):
                    self.friends[i][counti[i]] = j
                    paired.append((i,j))
                    counti[i] += 1

    def scale_vertex(self, scale_factor):
        for i in range(len(self.vertex_list)):
            mag = np.sqrt(sum(self.vertex_list[i]**2.0))
            if (mag < 1.0):
                self.vertex_list[i] *= scale_factor
                ds = abs(1.0 - np.sqrt(sum(self.vertex_list[i]**2.0)))
                while ds > 1e-12:
                    self.vertex_list[i] *= (ds + 1.0)
                    ds = abs(1.0 - np.sqrt(sum(self.vertex_list[i]**2.0)))

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2.0)
        b, c, d = -axis*np.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


    def twist_grid(self, angle=np.pi/5.0):
        axis = np.array([0.0, 0.0, 1.0])
        theta = angle

        for i in range(len(self.vertex_list)):
            if self.vertex_list[i][2] < 0.0:
                self.vertex_list[i] = np.dot(self.rotation_matrix(axis,theta), self.vertex_list[i])
            # print(np.dot(rotation_matrix(axis,theta), icosahedron.vertex_list[i]))

    def order_friends(self):
        def length(v):
            return sqrt(v[0]**2+v[1]**2)
        def dot_product(v,w):
           return v[0]*w[0]+v[1]*w[1]
        def determinant(v,w):
           return v[0]*w[1]-v[1]*w[0]
        def inner_angle(v,w):
           cosx=dot_product(v,w)/(length(v)*length(w))
           rad=arccos(cosx) # in radians
           return rad*180/np.pi # returns degrees
        def angle_clockwise(A, B):
            inner=inner_angle(A,B)
            det = determinant(A,B)
            if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
                return inner
            else: # if the det > 0 then A is immediately clockwise of B
                return 360-inner

        for i in range(len(self.vertex_list)):
            #Convert central point to spherical coords
            point1 = self.cart2sph(self.vertex_list[i])[1:]

            lat1 = point1[0]
            lon1 = point1[1]

            p0 = [0,1]

            friend_coords = []
            angles = []

            new_friends = []

            friend_num = 6
            if np.any(self.friends[i]) == -1:
                friend_num = 5

            pentagon = False
            for j in range(6):
                f_index = self.friends[i][j]

                if f_index != -1:
                    coords = self.cart2sph(self.vertex_list[f_index])[1:]
                    lat2 = coords[0]
                    lon2 = coords[1]

                    friend_coords.append(self.map_coords(lat1,lat2,lon1,lon2))

                    p1 = friend_coords[-1]

                    angles.append(angle_clockwise(p0, p1))
                else:
                    pentagon = True


            friends = []
            for j in np.argsort(angles):
                friends.append(int(self.friends[i][j]))

            if pentagon:
                friends.append(-1)

            self.friends[i] = friends


    def map_coords(self,lat1,lat2,lon1,lon2):
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        lon1 = radians(lon1)
        lon2 = radians(lon2)

        m = 2.0 * (1.0 + sin(lat2)*sin(lat1) + cos(lat1)*cos(lat2)*cos(lon2-lon1))

        x = m * cos(lat2) * sin(lon2 - lon1)

        y = m * (sin(lat2)*cos(lat1) - cos(lat2)*sin(lat1)*cos(lon2-lon1))

        return [x, y]

    def find_centers(self):
        self.centers = np.zeros((len(self.vertex_list),6,2))
        for i in range(len(self.vertex_list)):
            p1 = self.vertex_list[i]

            total = 6
            if self.friends[i][-1] < 0:
                total = 5

            centers_ = np.ones((6,2))*-1
            for j in range(total):
                p2 = self.vertex_list[int(self.friends[i][j])]
                p3 = self.vertex_list[int(self.friends[i][(j+1)%total])]

                center = (p1 + p2 + p3)/3.0

                mag = np.sqrt(sum(center**2))

                center = center/mag

                center = self.cart2sph(center)[1:]

                centers_[j] = center


            self.centers[i] = centers_

    def sph2map(self, lat1, lat2, lon1, lon2):
        def get_m(lat1, lat2, lon1, lon2):
            return 2.0 / (1.0 + sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2-lon1))

        r = 252

        m = get_m(lat1, lat2, lon1, lon2)

        x = m*r*cos(lat2) * sin(lon2-lon1)

        y = m * r * (sin(lat2)*cos(lat1) - cos(lat2)*sin(lat1)*cos(lon2-lon1))

        return x, y

    def map2sph(self, lat1, lon1, x, y):
        r = 252

        if abs(x) < 1e-10:
            x = 0.0
        if abs(y) < 1e-10:
            y = 0.0

        def get_m(x, y):
            return (4*r**2 + x**2 + y**2)/(4*r**2)


        m = get_m(x, y)

        x = x/(m*r)
        y = y/(m*r)

        B = 0.5*m - 1.0 - y * tan(lat1) - x * (1.0/cos(lat1)) * tan(lon1)
        B = B / ((cos(lat1)*tan(lon1)*sin(lon1)) + cos(lat1)*cos(lon1) + tan(lat1)*sin(lat1) * (sin(lon1)*tan(lon1) + cos(lon1)))

        C = (1.0/cos(lat1)) * y + x * tan(lat1)*tan(lon1) + B * tan(lat1) * (sin(lon1)*tan(lon1) + cos(lon1))

        new_lat = np.arcsin(C)
        new_lon = np.arccos(B/cos(new_lat))

        return(new_lat, new_lon)

        # print("LONS: ", np.degrees(lon1), np.degrees(new_lon), np.degrees(lon1 - new_lon))
        # print("LATS: ", np.degrees(lat1), np.degrees(new_lat), np.degrees(lat1 - new_lat))
        # print(np.degrees(lon1 - new_lon), np.degrees(lat1 - new_lat))


    def adjust_nodes_to_cv_centre(self):
        for i in range(len(self.vertex_list)):
            #Convert central point to spherical coords
            point1 = np.radians(self.cart2sph(self.vertex_list[i])[1:])

            lat1 = point1[0]
            lon1 = point1[1]

            [x1, y1, z1] = self.vertex_list[i]

            total = 6
            if self.friends[i][-1] < 0:
                total = 5

            cv_x = []
            cv_y = []
            cv_z = []
            area_total = 0
            for j in range(total):
                friend_id = self.friends[i][j]

                # point2 =  np.radians(self.cart2sph(self.vertex_list[friend_id])[1:])
                # print(self.centers[i][j])

                lat2 = np.radians(self.centers[i][j][0])
                lon2 = np.radians(self.centers[i][j][1])

                [x2, y2, z2] = self.sph2cart(1.0, np.degrees(lat2)-90.0, np.degrees(lon2))

                lat3 = np.radians(self.centers[i][(j+1)%total][0])
                lon3 = np.radians(self.centers[i][(j+1)%total][1])

                [x3, y3, z3] = self.sph2cart(1.0, np.degrees(lat3)-90.0, np.degrees(lon3))

                c = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2-lon1));
                a = arccos(sin(lat2) * sin(lat3) + cos(lat2) * cos(lat3) * cos(lon3-lon2));
                b = arccos(sin(lat3) * sin(lat1) + cos(lat3) * cos(lat1) * cos(lon1-lon3));

                A = arccos((cos(a) - cos(b)*cos(c))/(sin(b)*sin(c)));
                B = arccos((cos(b) - cos(a)*cos(c))/(sin(a)*sin(c)));
                C = arccos((cos(c) - cos(b)*cos(a))/(sin(b)*sin(a)));

                area = 1.0**2.0 * ((A + B + C) - np.pi);
                area_total += area

                x = (x1+x2+x3)/3.0
                y = (y1+y2+y3)/3.0
                z = (z1+z2+z3)/3.0
                L = np.sqrt(x**2.0 + y**2.0 + z**2.0)

                cv_x.append(area*x/L)
                cv_y.append(area*y/L)
                cv_z.append(area*z/L)

            Cx = sum(cv_x)/area_total
            Cy = sum(cv_y)/area_total
            Cz = sum(cv_z)/area_total

            # print(x1,Cx,'\t',y1,Cy,'\t',z1,Cz)
            #
            # print(self.cart2sph(self.vertex_list[i])[1:])

            self.vertex_list[i][0] = Cx
            self.vertex_list[i][1] = Cy
            self.vertex_list[i][2] = Cz

            # print(self.cart2sph(self.vertex_list[i])[1:])



                # x, y = self.sph2map(lat1,lat2,lon1,lon2)

                # cv_x.append(x)
                # cv_y.append(y)

                # self.map2sph(lat1,lon1,x,y)

            # A = 0
            # Cx = 0
            # Cy = 0
            # for j in range(total):
            #     j2 = (j+1)%total
            #     A = A + (cv_x[j]*cv_y[(j+1)%total] - cv_x[(j+1)%total]*cv_y[j])
            #     Cx = Cx + (cv_x[j] + cv_x[j2])*(cv_x[j]*cv_y[j2] - cv_x[j2]*cv_y[j])
            #     Cy = Cy + (cv_y[j] + cv_y[j2])*(cv_x[j]*cv_y[j2] - cv_x[j2]*cv_y[j])
            #
            # A = 0.5*abs(A)
            # Cx = Cx/(6*A)
            # Cy = Cy/(6*A)

            # nlat, nlon = self.map2sph(lat1,lon1, Cx, Cy)


            # print(Cx,Cy)
            # plt.scatter(cv_x, cv_y)
            # plt.plot(0,0,'k+')
            # plt.plot(Cx,Cy,'r+')
            #
            # plt.show()


    def haversine(self,lat1,lat2,lon1,lon2):

        A = np.sin(0.5*(lat2-lat1))**2
        B = np.cos(lat1)*np.cos(lat2)*np.sin(0.5*(lon2-lon1))**2

        dangle = 2.0*np.arcsin(np.sqrt(A+B))
        return dangle


    def find_arc_lengths(self):
        def haversine(lat1,lat2,lon1,lon2):

            A = np.sin(0.5*(lat2-lat1))**2
            B = np.cos(lat1)*np.cos(lat2)*np.sin(0.5*(lon2-lon1))**2

            dangle = 2.0*np.arcsin(np.sqrt(A+B))
            return dangle

        self.arc_lengths = np.ones((len(self.vertex_list), 6))*-1

        for i in range(len(self.vertex_list)):
            total = 6
            if self.friends[i][-1] < 0:
                total = 5


            for j in range(total):
                p1 = np.deg2rad(self.centers[i][j])
                p2 = np.deg2rad(self.centers[i][(j+1)%total])

                lat1 = p1[0]
                lat2 = p2[0]

                lon1 = p1[1]
                lon2 = p2[1]



                dangle = self.haversine(lat1,lat2,lon1,lon2)

                self.arc_lengths[i][j] = dangle

            #print(self.arc_lengths[i])

    def find_arc_midpoints(self):
        self.arc_mids = np.ones((len(self.vertex_list),6,2))*-1
        for i in range(len(self.vertex_list)):
            total = 6
            if self.friends[i][-1] < 0:
                total = 5


            for j in range(total):
                p1 = np.deg2rad(self.centers[i][j])
                p2 = np.deg2rad(self.centers[i][(j+1)%total])

                lat1 = p1[0]
                lat2 = p2[0]

                lon1 = p1[1]
                lon2 = p2[1]

                Bx = np.cos(lat2)*np.cos(lon2-lon1)
                By = np.cos(lat2)*np.sin(lon2-lon1)
                latm = np.arctan2(np.sin(lat1) + np.sin(lat2), np.sqrt((np.cos(lat1) + Bx)**2 + By**2))

                lonm = lon1 + np.arctan2(By, np.cos(lat1)+Bx)

                self.arc_mids[i][j] = np.rad2deg(np.array([latm,lonm]))


    def find_normals(self):
        self.normals = np.ones((len(self.vertex_list),6,2))*-1.0
        for i in range(len(self.vertex_list)):
            count = 0

            total = len(self.friends[i])

            if self.friends[i][-1] < 0:
                total = 5

            for j in range(total):
                p1 = np.deg2rad(self.centers[i][j])
                p2 = np.deg2rad(self.centers[i][(j+1)%total])

                lat1 = p1[0]
                lat2 = p2[0]
                lon1 = p1[1]
                lon2 = p2[1]

                dangle = self.arc_lengths[i][j]

                #print(dangle)
                #if i==1 and j==0:
                #    print(np.rad2deg(lat1),np.rad2deg(lon1),np.rad2deg(lat2),np.rad2deg(lon2))
                f = np.array([0.5-1e-4,0.5+1e-4])

                latf = np.zeros(2)
                lonf = np.zeros(2)

                for k in range(2):
                    A = np.sin((1.0-f[k])*dangle)/np.sin(dangle)
                    B = np.sin(f[k]*dangle)/np.sin(dangle)

                    x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2)*np.cos(lon2)
                    y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2)*np.sin(lon2)
                    z = A*np.sin(lat1) + B*np.sin(lat2)
                    latf[k] = np.arctan2(z,np.sqrt(x**2.0 + y**2.0))
                    lonf[k] = np.arctan2(y,x)

                mp1 = np.rad2deg(np.array([latf[0],lonf[0]]))
                mp2 = np.rad2deg(np.array([latf[1],lonf[1]]))

                dmp = mp2-mp1

                dlat = dmp[0]
                dlon = dmp[1]

                #if i==1 and j==0:
                #    print(mp1,mp2,dlat,dlon)

                self.normals[i][j] = np.array([dlon,-dlat])/np.sqrt(sum(dmp**2))

            #print(self.normals[i])





    def sph2cart(self,r,theta,phi):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return np.array([x, y, z])



    def cart2sph(self,coords):
        x = coords[0]
        y = coords[1]
        z = coords[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.pi*0.5 - np.arccos(z/r)
        lon = np.arctan2(y,x)


        if np.rad2deg(lon)+180.0 > 359.9:
            return np.array([r, np.rad2deg(lat), 0.0])
        return np.array([r, np.rad2deg(lat), np.rad2deg(lon)+180.0])


if __name__== '__main__':

    def cart2sph(x,y,z):
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.pi*0.5 - np.arccos(z/r)
        lon = np.arctan2(y,x)


        if np.rad2deg(lon)+180.0 > 359.9:
            return [r, np.rad2deg(lat), 0.0]
        return [r, np.rad2deg(lat), np.rad2deg(lon)+180.0]

    def sph2cart(r,theta,phi):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return np.array([x, y, z])





    import Shape2

    f = (1.0 + np.sqrt(5.0))/2.0

    icosahedron = Shape2.Shape()

    # Define the 12 vertices of an icosahedra

    r = 1.0

    v1 = sph2cart(r, 0.0, 0.)
    v2 = sph2cart(r, 180.0, 0.0)

    v3 = sph2cart(r,90.0-26.57, 36.0)
    v4 = sph2cart(r,90.0-26.57, 36.0*3.0)

    v5 = sph2cart(r,90.0-26.57, 36.0*5.0)
    v6 = sph2cart(r,90.0-26.57, 36.0*7.0)
    v7 = sph2cart(r,90.0-26.57, 36.0*9.0)
    v8 = sph2cart(r,(90.0+26.57), 0.0)

    v9 = sph2cart(r,(90.0+26.57), 36.0*2.0)
    v10 = sph2cart(r,(90.0+26.57), 36.0*4.0)
    v11 = sph2cart(r,(90.0+26.57), 36.0*6.0)
    v12 = sph2cart(r,(90.0+26.57), 36.0*8.0)



    icosahedron.add_vertex(v1)
    icosahedron.add_vertex(v2)
    icosahedron.add_vertex(v3)
    icosahedron.add_vertex(v4)

    icosahedron.add_vertex(v5)
    icosahedron.add_vertex(v6)
    icosahedron.add_vertex(v7)
    icosahedron.add_vertex(v8)

    icosahedron.add_vertex(v9)
    icosahedron.add_vertex(v10)
    icosahedron.add_vertex(v11)
    icosahedron.add_vertex(v12)

    print(icosahedron.vertex_list)

    scale_factor = 1.0 / np.sin(2*np.pi/5) / 2.0

    icosahedron.scale_vertex(scale_factor)

    icosahedron.find_friends()

    L = int(sys.argv[1])

    mins = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    for i in range(L-1):
        print("Calculating L" + str(i+1))
        icosahedron.bisect_edges()
        print("\tBisection complete...")
        icosahedron.scale_vertex(scale_factor)
        print("\tProjecting onto sphere...")
        # icosahedron.min_mag = mins[i+1]
        print("\tIdentifying neighbourghs...\n\n")
        if (i == L-2):
            print("\tRotating Southern Hemisphere by pi/2...")
            icosahedron.twist_grid()
        icosahedron.find_friends()

    print("Ordering neighbouring point in clockwise fashion...")
    icosahedron.order_friends()

    print("Finding triangular centroids...")
    icosahedron.find_centers()

    # icosahedron.adjust_nodes_to_cv_centre()


    print("Finding arc lengths between centroids...")
    icosahedron.find_arc_lengths()

    print("Finding arc length centers...")
    icosahedron.find_arc_midpoints()

    print("Finding normal vectors to cells...")
    icosahedron.find_normals()


    print("Calculations complete. Plotting...")
    m = Basemap(projection='ortho',lon_0=0,lat_0=0)
    lats = []
    lons = []
    for i in range(len(icosahedron.centers)):
        total = 6
        if icosahedron.centers[i][-1][0] < 0:
            total = 5
        for j in range(total):
            sph = icosahedron.centers[i][j]
            lats.append(sph[0])
            lons.append(sph[1])

    lats = np.array(lats)
    lons = np.array(lons)



    m = Basemap(projection='hammer',lon_0=180)
    m = Basemap(projection='ortho',lon_0=0,lat_0=90)
    x, y = m(lons,lats)
    # m.scatter(x,y,marker='o',s=8,color='k')
    # plt.show()

    lats = []
    lons = []
    for i in range(len(icosahedron.vertex_list)):
       sph = cart2sph(icosahedron.vertex_list[i][0], icosahedron.vertex_list[i][1], icosahedron.vertex_list[i][2])
       lats.append(sph[1])
       lons.append(sph[2])

    lats = np.array(lats)
    lons = np.array(lons)
    # for num in range(len(icosahedron.vertex_list)):
    #    for i in icosahedron.friends[num]:
    #         if i >= 0:
    #            m.drawgreatcircle(lons[num],lats[num],lons[int(i)],lats[int(i)],c='k',lw=0.5)
    #
    #
    #
    # #    if icosahedron.friends[num][-1] < 0:
    # #        total = 5
    #    #
    # #    for i in range(total):
    # #        sph1 = icosahedron.centers[num][i]
    # #        lat1 = sph1[0]
    # #        lon1 = sph1[1]
    #    #
    # #        sph2 = icosahedron.centers[num][(i+1)%total]
    # #        lat2 = sph2[0]
    # #        lon2 = sph2[1]
    #    #
    # #        m.drawgreatcircle(lon1, lat1, lon2, lat2, c='b', lw=0.4)
    #    #
    # #        #x, y = m(lon,lat)
    # #        #m.scatter(x, y, marker='o',s=2,color='k')
    # #    for i in range(total):
    # #        lat1 = icosahedron.arc_mids[num][i][0]
    # #        lon1 = icosahedron.arc_mids[num][i][1]
    #    #
    #    #
    # #        x, y = m(lon1,lat1)
    # #        m.scatter(x, y, marker='o',s=2,color='k')
    #
    #    print(icosahedron.normals[num])
    #
    # plt.show()

    # f = open('grid_l'+str(L)+'_testing.txt','w')
    # for i in range(len(icosahedron.vertex_list)):
    #     f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
    #     string = '{'
    #     for j in range(len(icosahedron.friends[0])):
    #        string += '{:3d}'.format(int(icosahedron.friends[i][j]))
    #        if j < len(icosahedron.friends[0]) - 1:
    #            string += ', '
    #        else:
    #            string += '}\n'
    #     f.write(string)
    # f.close() # you can omit in most cases as the destructor will call it

    #
    f = open('grid_l'+str(L)+'.txt','w')
    # f.write("ID     NODE_LAT      NODE_LON | FRIENDS LIST | CENTROID COORD LIST\n")
    f.write('{:<5s} {: <10s}   {: <10s}   {: <36s}  {:20s}'.format("ID", "NODE_LAT", "NODE_LON", "FRIENDS LIST", "CENTROID COORD LIST\n"))
    for i in range(len(icosahedron.vertex_list)):
        f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
        string = '{'
        for j in range(len(icosahedron.friends[0])):
           string += '{:4d}'.format(int(icosahedron.friends[i][j]))
           if j < len(icosahedron.friends[0]) - 1:
               string += ', '
           else:
               string += '}, '

        string += '{'
        for j in range(len(icosahedron.centers[0])):
            string += '({:10.6f}, {:10.6f})'.format(icosahedron.centers[i][j][0], icosahedron.centers[i][j][1])
            if j < len(icosahedron.centers[0]) - 1:
                string += ', '
            else:
                string += '} \n'
        # print(string)
        f.write(string)
    f.close() # you can omit in most cases as the destructor will call it
