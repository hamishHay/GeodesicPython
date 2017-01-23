
import numpy as np
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

                    self.vertex_list.append(vector)

        bset = set(tuple(x) for x in self.vertex_list)
        self.vertex_list = list(bset)

        for i in range(len(self.vertex_list)):
            self.vertex_list[i] = np.array(self.vertex_list[i])

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
            if i%100 == 0:
                print(float(i)/float(len(self.vertex_list))*100)
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
        for i in range(len(self.vertex_list)):
            point1 = self.cart2sph(self.vertex_list[i])[1:]

            pentagon = 0
            if (self.friends[i][-1] < 0):
                pentagon = 1

            lons = []
            for j in range(len(self.friends[i])-pentagon):
                lons.append(self.cart2sph(self.vertex_list[int(self.friends[i][j])])[2])

            transform_ind = np.zeros(6)
            if point1[1] < 30.0 and max(lons) > 300:
                for j in range(len(self.friends[i])-pentagon):
                    if (lons[j] > 300.0):
                        transform_ind[j] = -(360.0-lons[j])
            elif point1[1] > 300.0 and min(lons) < 30.0:
                for j in range(len(self.friends[i])-pentagon):
                    if (lons[j] < 300.0):
                        transform_ind[j] = 360.0 + lons[j]

            mag_north = 1.0
            north_vector = np.array([-1.0, 0.0])

            angles = []

            for j in range(len(self.friends[i])-pentagon):
                point2 = self.cart2sph(self.vertex_list[int(self.friends[i][j])])[1:]
                if transform_ind[j] != 0.0:
                    point2[1] = transform_ind[j]
                vector = point2 - point1
                dot = np.dot(north_vector,vector)
                det = north_vector[0]*vector[1] - north_vector[1]*vector[0]
                angle = np.arctan2(det,dot)

                angles.append(np.rad2deg(angle)+180.0)

            count = 0
            temp = []
            while count < len(angles):
                ind = np.argmin(angles)
                temp.append(int(self.friends[i][ind]))
                angles[ind] = 1000
                count+=1

            if pentagon:
                temp.append(-1)
            self.friends[i] = temp


    def find_normals(self):
        self.normals = np.ones((len(self.vertex_list),6,2))*-1.0
        for i in range(len(self.vertex_list)):
            count = 0

            total = len(self.friends[i])

            if self.friends[i][-1] < 0:
                total = 5

            while count < total:
                # pointm = self.cart2sph(self.vertex_list[i])[1:]
                point1 = self.cart2sph(self.vertex_list[int(self.friends[i][count])])[1:]
                point2 = self.cart2sph(self.vertex_list[int(self.friends[i][(count+1) % total])])[1:]

                vec = point2 - point1
                self.normals[i][count][0] = vec[1]
                self.normals[i][count][1] = -vec[0]
                self.normals[i][count] /= np.sqrt(sum(self.normals[i][count]**2))

                count+=1

            print(self.friends[i], self.normals[i])

    #def arclength(self):


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

    # axis = np.array([1.0, 0.0, 0.0])
    # theta = np.deg2rad(57.0)
    #
    # for i in range(len(icosahedron.vertex_list)):
    #     icosahedron.vertex_list[i] = np.dot(rotation_matrix(axis,theta), icosahedron.vertex_list[i])
    #     print(np.dot(rotation_matrix(axis,theta), icosahedron.vertex_list[i]))



    # icosahedron.min_mag = 2.0
    icosahedron.find_friends()

    L = 0

    mins = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    for i in range(L+1):
        print("Calculating L" + str(i+1))
        icosahedron.bisect_edges()
        print("\tBisection complete...")
        icosahedron.scale_vertex(scale_factor)
        print("\tProjecting onto sphere...")
        # icosahedron.min_mag = mins[i+1]
        print("\tIdentifying neighbourghs...\n\n")
        if (i == L):
            icosahedron.twist_grid()
        icosahedron.find_friends()

    icosahedron.order_friends()

    icosahedron.find_normals()

    print("Calculations complete. Plotting...")

    lats = []
    lons = []
    for i in range(len(icosahedron.vertex_list)):
        sph = cart2sph(icosahedron.vertex_list[i][0], icosahedron.vertex_list[i][1], icosahedron.vertex_list[i][2])
        lats.append(sph[1])
        lons.append(sph[2])

    lats = np.array(lats)
    lons = np.array(lons)


    # m = Basemap(projection='hammer',lon_0=180)
    m = Basemap(projection='ortho',lon_0=0,lat_0=0)
    x, y = m(lons,lats)
    # m.scatter(x,y,marker='o',s=8,color='k')
    # plt.show()
    #

    for num in range(len(lats)):
    # x, y = m(lons,lats)
        x20, y20 = m(lons[num],lats[num])
        # m.scatter(x20,y20,marker='o',s=6,color='r')

        for i in icosahedron.friends[num]:
            # for k in range(3):
            #     mag += (b[num][k] - b[int(i)][k])**2
            # mag = np.sqrt(mag)
            if i >= 0:# and (x20 != 0 or x20 <350)  and (x2 != 0 or x20 <350):
                x2, y2 = m(lons[i],lats[i])
                # m.plot([x20,x2],[y20,y2],color='k')
                # if (lons[num] != 0 and lons[i] < 350) or (lons[i] != 0 and lons[num] < 350):
                m.drawgreatcircle(lons[num],lats[num],lons[i],lats[i],c='k')
    plt.show()

    # f = open('grid_l'+str(L)+'_test.txt','w')
    # for i in range(len(icosahedron.vertex_list)):
    #     f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
    #     string = '{'
    #     for j in range(len(icosahedron.friends[0])):
    #         string += '{:3d}'.format(int(icosahedron.friends[i][j]))
    #         if j < len(icosahedron.friends[0]) - 1:
    #             string += ', '
    #         else:
    #             string += '}\n'
    #     f.write(string)
    # f.close() # you can omit in most cases as the destructor will call it
