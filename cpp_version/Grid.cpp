#include "Node.h"
#include "Grid.h"
#include "math_functions.h"

#include <iostream>
#include <vector>


Grid::Grid(void){};

void Grid::addNode(Node *n)
{
    node_list.push_back(n);
};

void Grid::findFriends(void)
{
    int i, j;
    double dist;
    Node * nodeA, * nodeB;

    for (i=0; i<node_list.size(); i++)
    {

        std::vector<int> surrounding_nodes;

        nodeA = this->node_list[i];
        // std::cout<<nodeA->ID<<'\t';
        for (j=0; j<node_list.size(); j++)
        {
            if (i != j)
            {
                nodeB = this->node_list[j];

                dist = (*nodeA - *nodeB).getMagnitude();

                if (dist < 1.6)
                {
                    surrounding_nodes.push_back( nodeB->ID );
                    nodeA->addFriend( nodeB );

                    // std::cout<<nodeB->ID<<'\t';
                }
            }
        }
        // std::cout<<std::endl;

        surrounding_nodes.push_back(-1);
        nodeA->friends_list.push_back(NULL);

        this->friends_list.push_back( surrounding_nodes );
    }
};

void Grid::bisectEdges(void)
{
    int node_count, i, j, k, added;
    double min_dist = 1.0;
    Node * node, *node_friend, *node_friend2;
    Node * inter_friend1, * inter_friend2;
    std::vector< Node * > new_nodes;

    node_count = this->node_list.size()-1;

    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
        // std::cout<<node->updated[0]<<std::endl;
        for (j=0; j<node->friend_num; j++)
        {
            Node * new_node;
            if (node->updated[j] == 0)
            {
                node_friend = node->friends_list[j];

                new_node = *(*node + *node_friend) * 0.5;

                node_count += 1;
                new_node->project2Sphere();
                new_node->ID = node_count;
                new_node->friend_num = 6;
                new_node->printCoords();

                min_dist = std::min(min_dist, new_node->getMagnitude());

                node->addTempFriend(new_node);
                new_nodes.push_back(new_node);

                new_node->addFriend(node);
                new_node->addFriend(node_friend);

                for (k=0; k<node_friend->friend_num; k++)
                {
                    node_friend2 = node_friend->friends_list[k];

                    if (*node == *node_friend2)
                    {
                        node_friend->addTempFriend(new_node);
                        node_friend->updated[k] = 1;
                        break;
                    }
                }

                node->updated[j] = 1;

            }
        }
    }

    // for (i=0; i<node_list.size(); i++)
    // {
    //     node = this->node_list[i];
    //
    //     added = 0;
    //     for (j=0; j<node->friend_num; j++)
    //     {
    //         inter_friend1 = node->temp_friends[j]
    //     }
    // }
}
