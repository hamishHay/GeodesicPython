#include "Node.h"
#include "Grid.h"
#include "math_functions.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>

struct AngleSort{
  double ang;
  Node * node;

  bool operator<( const AngleSort& rhs ) const { return ang < rhs.ang; }
};

Grid::Grid(void)
{
  this->recursion_lvl = 0;
};

void Grid::addNode(Node *n)
{
    node_list.push_back(n);
};

void Grid::findFriends(void)
{
    int i, j;
    double dist;
    Node * nodeA, * nodeB;
    Node * node, * node_friend;

    for (i=0; i<node_list.size(); i++)
    {

        std::vector<int> surrounding_nodes;

        nodeA = this->node_list[i];
        for (j=0; j<node_list.size(); j++)
        {
            if (i != j)
            {
                nodeB = this->node_list[j];

                dist = (*nodeA - *nodeB).getMagnitude();

                // std::cout<<dist<<std::endl;

                if (dist < 1.06)
                {
                    surrounding_nodes.push_back( nodeB->ID );
                    if (nodeA->friends_list.size()<5) nodeA->addFriend( nodeB );
                    // if (nodeB->friends_list.size()<5) nodeB->addFriend( nodeA );
                }
            }
        }

        surrounding_nodes.push_back(-1);
        nodeA->friends_list.push_back(NULL);

        this->friends_list.push_back( surrounding_nodes );
    }

    double dot_prod;
    double det;
    double mag1, mag2;
    double cosx, rad;
    double inner_angle;

    // std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
      //Order the friends of the new guys
      // double v1[2], v2[2];

      std::vector<AngleSort> ordered_friends(5);
      double v1[2] = {0., 1.};
      double v2[2];
      double angle;

      node = node_list[i];
      for (j=0; j<5; j++)
      {
        std::cout<<node->friends_list.size()<<std::endl;
        node_friend = node->friends_list[j];
        node_friend->getMapCoords(*node, v2);

        dot_prod = v1[0]*v2[0] + v1[1]*v2[1];

        det = v1[0]*v2[1] - v1[1]*v2[0];

        ordered_friends[j].ang = atan2(dot_prod, det)*180./pi;

        ordered_friends[j].node = node_friend;
      }

      std::sort(ordered_friends.begin(), ordered_friends.end());

      for (j=0; j<5; j++)
      {
        node->friends_list[j] = ordered_friends[j].node;
      }

    }
};

void Grid::bisectEdges(void)
{
    int node_count, i, j, k, added;
    double min_dist = 1.0;
    double dist;
    Node * node, *node_friend, *node_friend2;
    Node * inter_friend1, * inter_friend2;
    std::vector< Node * > new_nodes;

    this->recursion_lvl += 1;

    node_count = this->node_list.size()-1;

    std::cout<<'\t'<<"Beginning bisection..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
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

                min_dist = std::min(min_dist, (*node - *new_node).getMagnitude());

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
    std::cout<<"\r\tBeginning bisection... complete!"<<std::endl;

    std::cout<<'\t'<<"Determining all friends of new nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];

        for (j=0; j<node->friend_num; j++)
        {
            inter_friend1 = node->temp_friends[j];

            for (k=0; k<node->friend_num; k++)
            {
              inter_friend2 = node->temp_friends[k];

              if (*inter_friend1 != *inter_friend2)
              {
                dist = (*inter_friend1 - *inter_friend2).getMagnitude();

                if (dist < min_dist*1.2)
                {
                  inter_friend1->addFriend(inter_friend2);
                }
              }
            }
        }
    }
    std::cout<<'\r'<<"\tDetermining all friends of new nodes... complete!"<<std::endl;

    std::cout<<'\t'<<"Assigning new nodes to old nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
        for (j=0; j<node->friend_num; j++)
        {
            node->friends_list[j] = node->temp_friends[j];
            node->updated[j] = 0;
        }

        node->temp_friends.clear();
    }
    std::cout<<'\r'<<"\tAssigning new nodes to old nodes... complete!"<<std::endl;

    std::cout<<'\t'<<"Adding new nodes to master list..."<<std::flush;
    node_list.insert(node_list.end(),new_nodes.begin(),new_nodes.end());
    std::cout<<'\r'<<"\tAdding new nodes to master list... complete!"<<std::endl;

};

void Grid::orderFriends(void)
{
    int i, j, k;
    Node * node, *node_friend, *node_friend2;
    double v2[2];

    // std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
      //Order the friends of the new guys
      // double v1[2], v2[2];
      node = node_list[i];

      std::vector<AngleSort> ordered_friends(node->friend_num);


      for (j=0; j<node->friend_num; j++)
      {
        node_friend = node->friends_list[j];
        node_friend->getMapCoords(*node, v2);

        ordered_friends[j].ang = atan2(v2[0], v2[1])*180./pi;// - 90.0 + 360.0;

        ordered_friends[j].node = node_friend;
      }

      std::sort(ordered_friends.begin(), ordered_friends.end());

      for (j=0; j<node->friend_num; j++)
      {
        node->friends_list[j] = ordered_friends[j].node;
      }

    }
}

void Grid::findCentroids(void)
{
    int i, j, k;
    double mag;
    Node * node, *node_friend, *node_friend2;
    double xy_center[3];
    double sph_center[3];

    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
        for (j=0; j<node->friend_num; j++)
        {
            node_friend  = node->friends_list[j];
            node_friend2 = node->friends_list[(j+1)%node->friend_num];

            for (k=0; k<3; k++) {
                xy_center[k] = (node->xyz_coords[k]
                                + node_friend->xyz_coords[k]
                                + node_friend2->xyz_coords[k])/3.0;
            }

            mag = sqrt(xy_center[0]*xy_center[0] + xy_center[1]*xy_center[1] + xy_center[2]*xy_center[2]);
            xy_center[0] /= mag;
            xy_center[1] /= mag;
            xy_center[2] /= mag;

            cart2sph(xy_center, sph_center);

            node->centroids[j][0] = sph_center[0];
            node->centroids[j][1] = sph_center[1];
            node->centroids[j][2] = sph_center[2];

        }

        if (node->friend_num == 5)
        {
            node->centroids[5][0] = -1.0;
            node->centroids[5][1] = -1.0;
            node->centroids[5][2] = -1.0;
        }
    }
}

void Grid::saveGrid2File(void)
{
    FILE * outFile;
    Node * node;
    double lat, lon;
    int i, j;
    int f[6];
    double cx[6], cy[6];

    outFile = fopen("test_file.txt", "w");

    fprintf(outFile, "%-5s %-12s %-12s %-38s %-20s\n", "ID", "NODE LAT", "NODE LON", "FRIENDS LIST", "CENTROID COORD LIST");

    for (i=0; i<this->node_list.size(); i++)
    {
        node = this->node_list[i];
        lat = node->sph_coords[1]*180./pi;
        lon = node->sph_coords[2]*180./pi;
        for (j=0; j<node->friend_num; j++)
        {
            f[j] = node->friends_list[j]->ID;
            cx[j] = node->centroids[j][1]*180./pi;
            cy[j] = node->centroids[j][2]*180./pi;
        }
        if (node->friend_num == 5) {
            f[5] = -1;
            cx[j] = -1.0;
            cy[j] = -1.0;
        }
        fprintf(outFile, "%-5d %12.7f %12.7f { %4d, %4d, %4d, %4d, %4d, %4d}, {( % 10.7f, % 10.7f), ( % 10.7f, % 10.7f), ( % 10.7f, % 10.7f), ( % 10.7f, % 10.7f), ( % 10.7f, % 10.7f), ( % 10.7f, % 10.7f)} \n",
                i, lat, lon, f[0], f[1], f[2], f[3], f[4], f[5],
                cx[0], cy[0], cx[1], cy[1], cx[2], cy[2], cx[3], cy[3], cx[4], cy[4], cx[5], cy[5]);
    }


    fclose(outFile);
}
