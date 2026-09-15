import numpy as np 
import matplotlib.pyplot as plt


def sph2cart(point):
    r = point[0]
    lat = point[1]
    lon = point[2]

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.array([x, y, z])

def isInTriangle(p1, p2, p3, pt):
    p1_c = sph2cart( np.deg2rad(p1) )
    p2_c = sph2cart( np.deg2rad(p2) )
    p3_c = sph2cart( np.deg2rad(p3) )

    mat = np.zeros( (3, 3) )

    mat[:,0] = p1_c
    mat[:,1] = p2_c 
    mat[:,2] = p3_c 

    v = sph2cart( np.deg2rad(pt) )

    x = np.linalg.solve(mat, v)

    lam = np.sum(x)

    tol = -1e-16
    if (x[0] > tol and x[1] > tol and x[2] > tol and lam > tol): return True
    return False



# Define triangle 
p1 = np.array([1.0, 45.0, 20.])
p2 = np.array([1.0, 45.0, -20.])
p3 = np.array([1.0, 0.0, 0.0])


# xs = 
points = [p1, p2, p3]
# for point in points:
plt.plot([p1[2], p2[2], p3[2], p1[2]], [p1[1], p2[1], p3[1], p1[1]], '-o')

# print(points[:][1])

p1_c = sph2cart( np.deg2rad(p1) )
p2_c = sph2cart( np.deg2rad(p2) )
p3_c = sph2cart( np.deg2rad(p3) )

mat = np.zeros( (3, 3) )

mat[:,0] = p1_c
mat[:,1] = p2_c 
mat[:,2] = p3_c 

lats = 2*(np.random.rand(10000) - 0.5)*60.0
lons = 2*(np.random.rand(10000)- 0.5)*50.0

px = np.zeros(10000)
py = np.zeros(10000)
c_arr = []
for i in range(10000):
    pt = np.array([1.0, lats[i], lons[i]])
    c = (0.1, 0.3, 0.2)
    if isInTriangle(p1, p2, p3, pt): 
        c = (0.6, 0.3, 0.2)

    c_arr.append(c)
    # plt.plot(pt[2], pt[1], 'o', color=c)
    px[i] = pt[2]
    py[i] = pt[1]

plt.scatter(px, py, c=c_arr)
# pt = np.array([1.0, -20.0, 0])

# v = sph2cart( np.deg2rad(pt) )

# x = np.linalg.solve(mat, v)

# lam = np.sum(x)

# pt = p3.copy()
# print(isInTriangle(p1, p2, p3, pt))
# print(lats)
# print(x, lam)

plt.show()