
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Shape:
    def __init__(self):
        self.vertex_list = []
        self.face_list = []
        self.face_center = []

    def add_vertex(self, vertex):
        self.vertex_list.append(vertex)

    def print_vertex(self):
        for i in self.vertex_list:
            print(i.coords_cart)

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

    def calc_faces(self):
        self.center_max = 0
        for vertex1 in self.vertex_list:
            for vertex2 in self.vertex_list:
                if vertex1 == vertex2:
                    break
                for vertex3 in self.vertex_list:
                    if vertex2 == vertex3 or vertex1 == vertex3:
                        break
                    else:
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
                                                       vertex3))
                            self.face_center.append(point_center)
                            self.center_max = max(self.center_max,np.linalg.norm(point_center))

        rem_ind = []
        for i in range(len(self.face_list)):
            if np.linalg.norm(self.face_center[i]) < self.center_max*0.8:
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

class Face:
    def __init__(self,v1,v2,v3):
        self.vertices = [v1,v2,v3,v1]
        self.vx = np.array([v1.x,v2.x,v3.x,v1.x])
        self.vy = np.array([v1.y,v2.y,v3.y,v1.y])
        self.vz = np.array([v1.z,v2.z,v3.z,v1.z])


if __name__== '__main__':

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
    #
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for i in icosahedron.vertex_list:
    #     ax.scatter(i.coords_cart[0],i.coords_cart[1],i.coords_cart[2])
    #     plt.hold('on')

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    count = 0
    verts = []
    for i in icosahedron.face_list:
        print(icosahedron.face_center[count])
        v1 = [i.vx[0],i.vy[0],i.vz[0]]
        v2 = [i.vx[1],i.vy[1],i.vz[1]]
        v3 = [i.vx[2],i.vy[2],i.vz[2]]
        verts = [v1,v2,v3]
        polygon = Poly3DCollection([verts],alpha=0.5)
        face_color = [153/255.0,255/255.0,153/255.0]
        polygon.set_facecolor(face_color)
        polygon.set_alpha(0.5)
        ax.add_collection3d(polygon)
        # ax.scatter(icosahedron.face_center[count][0],
        #            icosahedron.face_center[count][1],
        #            icosahedron.face_center[count][2],
        #            marker='+',color='k')
        # plt.show()
        count += 1

    ax.set_aspect('equal')
    ax.set_xlim([1,-1])
    ax.set_ylim([1,-1])
    ax.set_zlim([1,-1])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    plt.axis('off')
    ax.set_axis_bgcolor(np.array([40.0,50.0,54.0])/255.0)
    fig.savefig('totally_worth_the_time.pdf')
    plt.show()
    plt.close()

