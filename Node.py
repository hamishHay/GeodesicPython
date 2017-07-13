import numpy as np

class Node:
    def __init__(self, x, y, z, ID):
        self.x = x
        self.y = y
        self.z = z
        self.ID = ID

        self.coords_cart = np.array([x,y,z])
        self.coords_sph = self.cart2sph(self.coords_cart)

        self.updated = [0, 0, 0, 0, 0, 0]
        self.friends = []
        self.inter_friends = [0, 0, 0, 0, 0, 0]
        self.f_num = 6

        self.pentagon = False

    def __add__(self, point):
        return self.coords_cart + point.coords_cart

    def __sub__(self, point):
        return self.coords_cart - point.coords_cart

    def add_friend(self, node):
        self.friends.append(node)

    def update_xyz(self, coords):
        self.coords_cart = coords
        self.coords_sph = self.cart2sph(self.coords_cart)

    def sph2cart(self,coords):
        r = coords[0]
        theta = coords[1]
        phi = coords[2]

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
            return np.array([r, lat, 0.0])
        return np.array([r, lat, lon+np.pi])
