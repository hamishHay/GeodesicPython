import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import sys

#grid = np.loadtxt('grid_l4.txt').T

load_file = "grid_l" + str(int(sys.argv[1])) + ".txt"

f = open(load_file,'r')
lines = f.readlines()

# [print(i) for i in lines]

lats = []
lons = []
friends = []
for line in lines[1:]:
    all = []
    line = line.split()


    lats.append(float(line[1]))
    lons.append(float(line[2]))

    if line[3] == "{":
        line.remove("{")

    start = 3
    for i in range(6):
        line[start+i] = line[start+i].strip('{')
        line[start+i] = line[start+i].strip('}')
        line[start+i] = line[start+i].strip(',')
        line[start+i] = line[start+i].strip('},')

        print(line)
        all.append(int(line[start+i]))
    friends.append(all)
    # break
f.close()

lats = np.array(lats)
lons = np.array(lons)
friends = np.array(friends)
data = np.cos(3*np.deg2rad(lats))*np.cos(np.deg2rad(lons))

plt.scatter(lons,lats,c='b',s=0.5)
plt.show()
ax = plt.gca()
ax.set_aspect('equal')

# # m = Basemap(projection='ortho',lon_0=-105,lat_0=40)
# m = Basemap(projection='hammer',lon_0=180)
m = Basemap(projection='ortho',lon_0=0,lat_0=90)
x, y = m(lons,lats)
# m.scatter(x,y,marker='o',s=2,color='k')

for num in range(len(lats)):
# x, y = m(lons,lats)
    x20, y20 = m(lons[num],lats[num])
    m.scatter(x20,y20,marker='o',s=6,color='r')

    for i in friends[num]:
        # for k in range(3):
        #     mag += (b[num][k] - b[int(i)][k])**2
        # mag = np.sqrt(mag)
        if i >= 0:# and (x20 != 0 or x20 <350)  and (x2 != 0 or x20 <350):
            x2, y2 = m(lons[i],lats[i])
            # m.plot([x20,x2],[y20,y2],color='k')
            # if (lons[num] != 0 and lons[i] < 350) or (lons[i] != 0 and lons[num] < 350):
            # m.drawgreatcircle(lons[num],lats[num],lons[i],lats[i],c='k')
    # plt.show()
# m.pcolor(x,y,data,tri=True,cmap=plt.cm.hot)
# plt.savefig("COOOOL.png",bbox_inches='tight')
plt.show()
