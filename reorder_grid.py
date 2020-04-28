import matplotlib as mpl
import ReadGrid
import numpy as np
from numpy import deg2rad
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from shapely.geometry import Polygon
from scipy.sparse.csgraph import reverse_cuthill_mckee

N = int(sys.argv[1])        # geodesic grid recursion level

# Load geodesic grid corresponding to level N
Grid = ReadGrid.read_grid(N)
num_nodes = len(Grid.nodes)
node_list = Grid.nodes

# CooMat = coo_matrix((num_nodes, num_nodes), np.int)

vals = np.zeros(6*12 + 7*(num_nodes-12), np.int)
cols = np.zeros(6*12 + 7*(num_nodes-12), np.int)
rows = np.zeros(6*12 + 7*(num_nodes-12), np.int)
count = 0
for i in range(num_nodes):
    node_central = node_list[i]
    friend_list = node_central.friends

    pent = 0
    if friend_list[-1] < 0:
        pent = 1


    # CooMat[i,i] = node_central.ID+1
    vals[count] = node_central.ID+1
    rows[count] = i
    cols[count] = i
    count+=1
    for j in range(6-pent):
        node_friend = node_list[friend_list[j]]

        # CooMat[i, node_friend.ID] = node_friend.ID+1

        vals[count] = node_friend.ID+1
        rows[count] = i
        cols[count] = node_friend.ID
        count+=1

CooMat = coo_matrix((vals, (rows, cols)), dtype=np.int)
SparseMat = csr_matrix(CooMat)
SP3 = SparseMat*SparseMat
rows, cols = np.nonzero(SP3)

LilMat = lil_matrix(CooMat)
for i in range(len(rows)):
    LilMat[rows[i], cols[i]] = cols[i]+1

SparseMat = csr_matrix(LilMat)


# print()
del CooMat
del LilMat

print("Computing new order....")
perm = reverse_cuthill_mckee(SparseMat)

print("Getting new order...")
new_nodes = perm[:]

print("Reordering grid...")
Grid.reorder(list(new_nodes))

print("Saving grid...")
Grid.save_grid()
