
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from scipy.stats import mode

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
                        vector.append((p2[k]-p1[k])*0.5)
                    vector = np.array(vector)

                    self.vertex_list.append(p1 + vector)

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
        # self.get_min_mag()
        paired = []
        counti = np.zeros(len(self.vertex_list))
        self.friends = np.ones((len(self.vertex_list), 6))*-1
        for i in range(len(self.vertex_list)):
            p1 = self.vertex_list[i]
            for j in range(len(self.vertex_list)):
                # if self.friends[i].any() < 0:
                #     break
                # else:
                p2 = self.vertex_list[j]
                mag = np.sqrt(sum((p1-p2)**2.0))
                # print(mag)
                if (mag <= self.min_mag*level and mag > 0.0):
                    # print(mag)
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


if __name__== '__main__':

    def cart2sph(x,y,z):
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.pi*0.5 - np.arccos(z/r)
        lon = np.arctan2(y,x)


        if np.rad2deg(lon)+180.0 > 359.9:
            return [r, np.rad2deg(lat), 0.0]
        return [r, np.rad2deg(lat), np.rad2deg(lon)+180.0]

    import Shape2

    f = (1.0 + np.sqrt(5.0))/2.0

    icosahedron = Shape2.Shape()

    # Define the 12 vertices of an icosahedra
    v1 = np.array([0, 1.0, f])
    v2 = np.array([0, -1.0, f])
    v3 = np.array([0, 1.0, -f])
    v4 = np.array([0, -1.0, -f])

    v5 = np.array([1.0, f, 0])
    v6 = np.array([-1.0, f, 0])
    v7 = np.array([1.0, -f, 0])
    v8 = np.array([-1.0, -f, 0])

    v9 = np.array([f, 0, 1.0])
    v10 = np.array([f, 0, -1.0])
    v11 = np.array([-f, 0, 1.0])
    v12 = np.array([-f, 0, -1.0])

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
    # icosahedron.min_mag = 2.0
    icosahedron.find_friends()


    mins = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    for i in range(5):
        print("Calculating L" + str(i+1))
        icosahedron.bisect_edges()
        print("\tBisection complete...")
        icosahedron.scale_vertex(scale_factor)
        print("\tProjecting onto sphere...")
        icosahedron.min_mag = mins[i+1]
        print("\tIdentifying neighbourghs...\n\n")
        icosahedron.find_friends()

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
    m = Basemap(projection='ortho',lon_0=-105,lat_0=40)
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
    #
    # # num = 5
    # # x20, y20 = m(lons[num],lats[num])
    # #
    # # num = 12
    # # x2, y2 = m(lons[num],lats[num])
    # # m.drawgreatcircle(lons[5],lats[5],lons[8],lats[8])
    #
    # # ax.scatter(lons,lats,c='b')
    # #
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
    # #
    # #
    # # num = 97
    # # ax.scatter(lons[num],lats[num],c='pink')
    # # for i in friends[num]:
    # #     ax.scatter(lons[i],lats[i],c='pink')
    # #
    # #
    # # num = 112
    # # ax.scatter(lons[num],lats[num],c='cyan')
    # # for i in friends[num]:
    # #     if i >= 0:
    # #         ax.scatter(lons[i],lats[i],c='cyan')
    # #
    # #
    # # ax.set_xlim([0,360])
    # # ax.set_ylim([90,-90])
    # # plt.show()
    # # # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # # # count = 0
    # # # verts = []
    # # # for i in icosahedron.face_list:
    # # #     v1 = [i.vx[0],i.vy[0],i.vz[0]]
    # # #     v2 = [i.vx[1],i.vy[1],i.vz[1]]
    # # #     v3 = [i.vx[2],i.vy[2],i.vz[2]]
    # # #     verts = [v1,v2,v3]
    # # #     polygon = Poly3DCollection([verts],alpha=0.5)
    # # #     face_color = np.array([153,255,153])/255.0
    # # #     polygon.set_facecolor(face_color)
    # # #     polygon.set_alpha(0.5)
    # # #     ax.add_collection3d(polygon)
    # # #     # ax.scatter(icosahedron.face_center[count][0],
    # # #     #            icosahedron.face_center[count][1],
    # # #     #            icosahedron.face_center[count][2],
    # # #     #            marker='+',color='k')
    # # #     # for j in range(3):
    # # #     #     ax.scatter(i.x_halves[j],
    # # #     #                i.y_halves[j],
    # # #     #                i.z_halves[j],
    # # #     #                marker='+',color='k')
    # # #     # plt.show()
    # # #     count += 1
    # # # # for i in icosahedron.vertex_list:
    # # # #     ax.scatter(i.coords_cart[0],i.coords_cart[1],i.coords_cart[2])
    # # # #     plt.hold('on')
    # # #
    # # # ax.set_aspect('equal')
    # # # ax.set_xlim([1,-1])
    # # # ax.set_ylim([1,-1])
    # # # ax.set_zlim([1,-1])
    # # # # ax.set_xlabel('x')
    # # # # ax.set_ylabel('y')
    # # # # ax.set_zlabel('z')
    # # # plt.axis('off')
    # # # ax.set_axis_bgcolor(np.array([40.0,50.0,54.0])/255.0)
    # # # fig.savefig('icosahedron.pdf',bbox_inches='tight')
    # # # plt.show()
    # # # plt.close()
    # #
    # #
    # #write file?
    f = open('grid_l5_test.txt','w')
    for i in range(len(icosahedron.vertex_list)):
        f.write('{:<5d} {: >10.6f}   {: >10.6f}   '.format(i, lats[i], lons[i])) # python will convert \n to os.linesep
        string = '{'
        for j in range(len(icosahedron.friends[0])):
            string += '{:3d}'.format(int(icosahedron.friends[i][j]))
            if j < len(icosahedron.friends[0]) - 1:
                string += ', '
            else:
                string += '}\n'
        f.write(string)
    f.close() # you can omit in most cases as the destructor will call it