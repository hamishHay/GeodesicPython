
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from scipy.stats import mode

class Shape:
    def __init__(self):
        self.vertex_list = []
        self.face_list = []
        self.face_center = []
        self.edge_len = 0

    def add_vertex(self, vertex):
        self.vertex_list.append(vertex)

    def print_vertex(self):
        for i in self.vertex_list:
            print(i.coords_cart)

    def set_edge_len(self,val):
        self.edge_len = val

    def scale_vertex(self,scale_factor):
        for i in self.vertex_list:
            i.scale_coords(scale_factor)

    def get_coord_lists(self):
        xlist = []
        ylist = []
        zlist = []

        for i in self.vertex_list:
            xlist.append(i.coords_cart[0])
            ylist.append(i.coords_cart[1])
            zlist.append(i.coords_cart[2])

        return xlist, ylist, zlist

    def remove_duplicates(self):
        del_list = []
        count = 0
        print(len(self.vertex_list))
        for i in range(len(self.vertex_list)):
            count += 1
            for j in range(len(self.vertex_list)):
                if i != j:
                    if self.vertex_list[i].coords_cart[0] == self.vertex_list[j].coords_cart[0] and self.vertex_list[i].coords_cart[1] == self.vertex_list[j].coords_cart[1] and self.vertex_list[i].coords_cart[2] == self.vertex_list[j].coords_cart[2]:
                        print(i,j,self.vertex_list[i])
                        if self.vertex_list[i].coords_cart[0] == 0 and self.vertex_list[i].coords_cart[1] == 0 and self.vertex_list[i].coords_cart[2] == 1:
                            print("Mun")
                        del_list.append(j)

        del_list = np.unique(del_list)[::-1]

        for i in del_list:
            del self.vertex_list[i]

    def bisect_edges(self):
        min_len = []
        for i in self.face_list:
            v, edge = i.bisect_edges()
            min_len.append(edge)
            for vertex in v:
                self.add_vertex(vertex)

        self.set_edge_len(min(min_len))
        print(len(self.vertex_list))

    def vec_length(self,v):
        return np.sqrt(sum(v**2))


    def calc_faces(self):
        self.center_max = 0
        self.face_list = []
        self.face_center = []

        print(len(self.vertex_list))
        if self.edge_len != 0 or self.edge_len==0:

            for vertex1 in self.vertex_list:
                for vertex2 in self.vertex_list:
                    length = np.sqrt(sum((vertex2.coords_cart - vertex1.coords_cart)**2))
                    if length > 0:
                        if self.edge_len == 0:
                            self.edge_len = length
                        self.edge_len = min(self.edge_len,length)

            print("edge len:",self.edge_len)
            factor = 1.3
            for vertex1 in self.vertex_list:
                for vertex2 in self.vertex_list:
                    vec1 = vertex2.coords_cart - vertex1.coords_cart
                    len1 = self.vec_length(vec1)
                    # if len1 == 0:# or len1 > self.edge_len:
                    #     break
                    if vertex1 != vertex2 and len1 <= self.edge_len*factor:
                        for vertex3 in self.vertex_list:
                            vec2 = vertex3.coords_cart - vertex2.coords_cart
                            len2 = self.vec_length(vec2)
                            if vertex1 != vertex3 and len2 <= self.edge_len*factor and vertex2 != vertex3 and len2 > 0:

                                # if len2 == 0:# or len2 > self.edge_len:
                                #     break

                                vec3 = vertex3.coords_cart - vertex1.coords_cart
                                len3 = self.vec_length(vec3)



                                # if len3 == 0:
                                #     break
                                # factor = 1.0001
                                # print(len1,len2,len3)
                                if len1 < self.edge_len*factor and len1> 0 and len2 < self.edge_len*factor and len2 > 0 and len3 < self.edge_len*factor and len3 > 0:


                                    point_center = np.array([np.mean([vertex1.coords_cart[0],
                                                                     vertex2.coords_cart[0],
                                                                     vertex3.coords_cart[0]]),
                                                             np.mean([vertex1.coords_cart[1],
                                                                     vertex2.coords_cart[1],
                                                                     vertex3.coords_cart[1]]),
                                                             np.mean([vertex1.coords_cart[2],
                                                                     vertex2.coords_cart[2],
                                                                     vertex3.coords_cart[2]])])

                                    if len(self.face_center) == 0:
                                        self.face_list.append(Face(vertex1,
                                                                       vertex2,
                                                                       vertex3,
                                                                       self))
                                        self.face_center.append(point_center)

                                    duplicate = False
                                    for i in self.face_center:
                                        if abs(i[0] - point_center[0]) + abs(i[1] - point_center[1]) + abs(i[2] - point_center[2]) < 1e-5:
                                            duplicate = True

                                    if not duplicate:
                                        # print("added face:", len(self.face_list))
                                        self.face_list.append(Face(vertex1,
                                                                   vertex2,
                                                                   vertex3,
                                                                   self))
                                        self.face_center.append(point_center)

        else:
            for vertex1 in self.vertex_list:
                for vertex2 in self.vertex_list:
                    if vertex1 == vertex2:
                        break
                    for vertex3 in self.vertex_list:
                        len1 = np.linalg.norm(vertex2.coords_cart - vertex1.coords_cart)
                        len2 = np.linalg.norm(vertex3.coords_cart - vertex1.coords_cart)

                        if vertex2 == vertex3 or vertex1 == vertex3:
                            break


                        else:
                            #

                            point_center = np.array([np.mean([vertex1.coords_cart[0],
                                                             vertex2.coords_cart[0],
                                                             vertex3.coords_cart[0]]),
                                                     np.mean([vertex1.coords_cart[1],
                                                             vertex2.coords_cart[1],
                                                             vertex3.coords_cart[1]]),
                                                     np.mean([vertex1.coords_cart[2],
                                                             vertex2.coords_cart[2],
                                                             vertex3.coords_cart[2]])])
                            normal_vector1 = vertex2.coords_cart - vertex1.coords_cart
                            normal_vector2 = vertex3.coords_cart - vertex1.coords_cart

                            normal_vector = np.cross(normal_vector1,normal_vector2)
                            cross_prod = np.cross(point_center,normal_vector)

                            if np.mean(abs(cross_prod)) < 1e-15:
                                self.face_list.append(Face(vertex1,
                                                           vertex2,
                                                           vertex3,
                                                           self))
                                self.face_center.append(point_center)
                                self.center_max = max(self.center_max,np.linalg.norm(point_center))

            # print(self.face_list)
            rem_ind = []
            for i in range(len(self.face_list)):
                if np.linalg.norm(self.face_center[i]) < self.center_max:
                    rem_ind.append(i)

            rem_ind =  np.sort(rem_ind)[::-1]

            for i in rem_ind:
                del self.face_list[i]
                del self.face_center[i]


class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.ID = 0

        self.coords_cart = np.array([self.x, self.y, self.z])

    def convert2spherical(self):
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.lon = np.arccos(self.z/self.r)
        self.colat = np.arctan(self.y/self.x)

        self.coords_spher = np.array([self.r,self.lon,self.colat])

    def scale_coords(self, scale_factor):
        self.coords_cart *= scale_factor
        self.x = self.coords_cart[0]
        self.y = self.coords_cart[1]
        self.z = self.coords_cart[2]

        ds = abs(1.0 - np.sqrt(sum(self.coords_cart**2)))
        # print(ds)
        while ds > 1e-12:
            self.coords_cart *= (ds + 1.0)
            self.x = self.coords_cart[0]
            self.y = self.coords_cart[1]
            self.z = self.coords_cart[2]

            ds = abs(1.0 - np.sqrt(sum(self.coords_cart**2)))



class Face:
    def __init__(self,v1,v2,v3,shape):
        self.vertices = [v1,v2,v3,v1]
        self.vx = np.array([v1.x,v2.x,v3.x,v1.x])
        self.vy = np.array([v1.y,v2.y,v3.y,v1.y])
        self.vz = np.array([v1.z,v2.z,v3.z,v1.z])
        self.parent_shape = shape

    def bisect_edges(self):
        self.x_halves = []
        self.y_halves = []
        self.z_halves = []
        for i in range(0,3):
            self.x_halves.append(0.5*(self.vx[i+1] + self.vx[i]))
            self.y_halves.append(0.5*(self.vy[i+1] + self.vy[i]))
            self.z_halves.append(0.5*(self.vz[i+1] + self.vz[i]))

        m = []
        v = []
        for i in range(len(self.x_halves)):
            m.append(np.array([self.x_halves[i], self.y_halves[i], self.z_halves[i]]))
            while 1 - np.linalg.norm(m[i]) > 1e-15:
                length = 2.0 - np.linalg.norm(m[i])
                self.x_halves[i] =  self.x_halves[i] * length
                self.y_halves[i] =  self.y_halves[i] * length
                self.z_halves[i] =  self.z_halves[i] * length
                m[-1] = np.array([self.x_halves[i], self.y_halves[i], self.z_halves[i]])

            v.append(Vertex(self.x_halves[i],
                       self.y_halves[i],
                       self.z_halves[i]))
        return v, length


if __name__== '__main__':

    def cart2sph(x,y,z):
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.pi*0.5 - np.arccos(z/r)
        lon = np.arctan2(y,x)


        if np.rad2deg(lon)+180.0 > 359.9:
            return [r, np.rad2deg(lat), 0.0]
        return [r, np.rad2deg(lat), np.rad2deg(lon)+180.0]

    import Shape

    f = (1.0 + np.sqrt(5.0))/2.0

    icosahedron = Shape.Shape()

    # Define the 12 vertices of an icosahedra
    v1 = Shape.Vertex(0, 1.0, f)
    v2 = Shape.Vertex(0, -1.0, f)
    v3 = Shape.Vertex(0, 1.0, -f)
    v4 = Shape.Vertex(0, -1.0, -f)

    v5 = Shape.Vertex(1.0, f, 0)
    v6 = Shape.Vertex(-1.0, f, 0)
    v7 = Shape.Vertex(1.0, -f, 0)
    v8 = Shape.Vertex(-1.0, -f, 0)

    v9 = Shape.Vertex(f, 0, 1.0)
    v10 = Shape.Vertex(f, 0, -1.0)
    v11 = Shape.Vertex(-f, 0, 1.0)
    v12 = Shape.Vertex(-f, 0, -1.0)

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

    scale_factor = 1.0 / np.sin(2*np.pi/5) / 2.0

    icosahedron.scale_vertex(scale_factor)

    icosahedron.calc_faces()


    for i in range(0):
        icosahedron.bisect_edges()
        icosahedron.scale_vertex(scale_factor)
        icosahedron.calc_faces()

    icosahedron.bisect_edges()
    icosahedron.calc_faces()
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal')
    # ax.set_xlim([1,-1])
    # ax.set_ylim([1,-1])
    # ax.set_zlim([1,-1])
    plt.show(block=False)

    # for i in icosahedron.vertex_list:
    #     # ax.scatter(i.coords_cart[0],i.coords_cart[1],i.coords_cart[2])
    #     print(np.sqrt(i.coords_cart[0]**2 + i.coords_cart[1]**2 + i.coords_cart[2]**2))
        # plt.hold('on')
    # plt.show()

    # print(icosahedron.vertex_list)

    all_vertices = []
    for i in icosahedron.face_list:
        for j in range(3):
            all_vertices.append([i.vx[j],i.vy[j],i.vz[j]])

    bset = set(tuple(x) for x in all_vertices)
    # b = [list(x) for x in bset]
    #
    # lats = []
    # lons = []
    # for i in range(len(b)):
    #     sph = cart2sph(b[i][0], b[i][1], b[i][2])
    #     lats.append(sph[1])
    #     lons.append(sph[2])
    #
    # lats = np.array(lats)
    # lons = np.array(lons)
    #
    #
    # mag = []
    # for coord in b[1:]:
    #     mag.append(0)
    #     for i in range(3):
    #         mag[-1] += (b[0][i]-coord[i])**2
    #     mag[-1] = np.sqrt(mag[-1])
    #
    # mag_max = min(mag)
    #
    # print("MAX: ", mag_max)
    #
    # friends = np.ones((len(b), 6))*-1
    #
    # mag = 0
    # counti = np.zeros(len(b))
    # for i in range(len(b)):
    #     for j in range(len(b)):
    #         for k in range(3):
    #             mag += (b[i][k] - b[j][k])**2
    #         mag = np.sqrt(mag)
    #
    #         print(mag)
    #         if mag > 0 and mag < mag_max*1.2:
    #             friends[i][counti[i]] = j
    #             counti[i] += 1
    #
    #             isph = cart2sph(b[i][0], b[i][1], b[i][2])
    #             jsph = cart2sph(b[j][0], b[j][1], b[j][2])
    #
    #         mag = 0

    # print(friends)

    # m = Basemap(projection='hammer',lon_0=180)
    # x, y = m(lons,lats)
    # m.scatter(x,y,color='b')
    # plt.show()


    # m = Basemap(projection='hammer',lon_0=180)
    # m = Basemap(projection='ortho',lon_0=-105,lat_0=40)
    # x, y = m(lons,lats)
    # m.scatter(x,y,marker='o',s=2,color='k')

    # num = 5
    # x20, y20 = m(lons[num],lats[num])
    #
    # num = 12
    # x2, y2 = m(lons[num],lats[num])
    # m.drawgreatcircle(lons[5],lats[5],lons[8],lats[8])

    # ax.scatter(lons,lats,c='b')
    #
    # for num in range(len(b)):
    # # x, y = m(lons,lats)
    #     x20, y20 = m(lons[num],lats[num])
    #     # m.scatter(x20,y20,marker='o',s=6,color='r')
    #
    #     mag = 0
    #     for i in friends[num]:
    #         # for k in range(3):
    #         #     mag += (b[num][k] - b[int(i)][k])**2
    #         # mag = np.sqrt(mag)
    #         if i >= 0:# and (x20 != 0 or x20 <350)  and (x2 != 0 or x20 <350):
    #             x2, y2 = m(lons[i],lats[i])
    #             # m.plot([x20,x2],[y20,y2],color='k')
    #             # if (lons[num] != 0 and lons[i] < 350) or (lons[i] != 0 and lons[num] < 350):
    #             m.drawgreatcircle(lons[num],lats[num],lons[i],lats[i],c='k')
    #
    # plt.show()

    #
    #
    # num = 97
    # ax.scatter(lons[num],lats[num],c='pink')
    # for i in friends[num]:
    #     ax.scatter(lons[i],lats[i],c='pink')
    #
    #
    # num = 112
    # ax.scatter(lons[num],lats[num],c='cyan')
    # for i in friends[num]:
    #     if i >= 0:
    #         ax.scatter(lons[i],lats[i],c='cyan')
    #
    #
    # ax.set_xlim([0,360])
    # ax.set_ylim([90,-90])
    # plt.show()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    count = 0
    verts = []
    for i in icosahedron.face_list:
        v1 = [i.vx[0],i.vy[0],i.vz[0]]
        v2 = [i.vx[1],i.vy[1],i.vz[1]]
        v3 = [i.vx[2],i.vy[2],i.vz[2]]
        verts = [v1,v2,v3]
        polygon = Poly3DCollection([verts],alpha=1.0)
        face_color = np.array([153,255,153])/255.0
        polygon.set_facecolor(face_color)
        polygon.set_alpha(0.5)
        ax.add_collection3d(polygon)
        # ax.scatter(icosahedron.face_center[count][0],
        #            icosahedron.face_center[count][1],
        #            icosahedron.face_center[count][2],
        #            marker='+',color='k')
        # for j in range(3):
        #     ax.scatter(i.x_halves[j],
        #                i.y_halves[j],
        #                i.z_halves[j],
        #                marker='+',color='k')
    # plt.show()
    # #     count += 1
    # # # for i in icosahedron.vertex_list:
    # # #     ax.scatter(i.coords_cart[0],i.coords_cart[1],i.coords_cart[2])
    # # #     plt.hold('on')
    # #
    ax.set_aspect('equal')
    ax.set_xlim([1,-1])
    ax.set_ylim([1,-1])
    ax.set_zlim([1,-1])
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    # plt.axis('off')
    # ax.set_axis_bgcolor(np.array([40.0,50.0,54.0])/255.0)
    # # fig.savefig('icosahedron.pdf',bbox_inches='tight')

    ax.view_init(elev=60.0, azim=0)

    plt.show()
    # plt.close()
    #
    #
    # #write file?
    f = open('myfile.txt','w')
    for i in range(len(b)):
        f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
        string = '{'
        for j in range(len(friends[0])):
            string += '{:3d}'.format(int(friends[i][j]))
            if j < len(friends[0]) - 1:
                string += ', '
            else:
                string += '}\n'
        f.write(string)
    f.close() # you can omit in most cases as the destructor will call it