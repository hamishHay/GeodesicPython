#include "Node.h"
#include "Grid.h"

#include <iostream>

Grid::Grid(void){};

void Grid::addNode(Node &n)
{
    nodes.push_back(n);

    std::cout<<"Adding node: "<<n.xyz_coords[0]<<'\t'<<n.xyz_coords[1]<<'\t'<<n.xyz_coords[2]<<std::endl;
};
