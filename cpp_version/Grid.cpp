#include "Node.h"
#include "Grid.h"

#include <iostream>

Grid::Grid(void){};

void Grid::addNode(Node &n)
{
    nodes.push_back(n);

    std::cout<<"Adding node: "<<n.xyz_coords[0]<<'\t'<<n.xyz_coords[1]<<'\t'<<n.xyz_coords[2]<<std::endl;
};

void Grid::findFriends(void)
{
    int i, j;
    double dist;

    for (i=0; i<nodes.size(); i++)
    {

        std::vector<int> surrounding_nodes;

        Node nodeA = this->nodes[i];
        for (j=0; j<nodes.size(); j++)
        {
            if (i != j)
            {
                Node nodeB = this->nodes[j];

                dist = (nodeA - nodeB).getMagnitude();

                if (dist < 1.6)
                {
                    surrounding_nodes.push_back( nodeB.ID );
                }
            }
        }

        surrounding_nodes.push_back(-1);

        this->friends_list.push_back( surrounding_nodes );
    }
};
