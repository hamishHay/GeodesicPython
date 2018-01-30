#include "Node.h"
#include "Grid.h"
#include "math_functions.h"

#include <iostream>
#include <vector>
#include <algorithm>

struct AngleSort{
  double ang;
  Node * node;

  bool operator<( const AngleSort& rhs ) const { return ang < rhs.ang; }
};

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

                if (dist < 1.6)
                {
                    surrounding_nodes.push_back( nodeB->ID );
                    nodeA->addFriend( nodeB );
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

    for (i=0; i<node_list.size(); i++)
    {
      //Order the friends of the new guys
      // double v1[2], v2[2];

      std::vector<double> angles(5);
      std::vector<AngleSort> ordered_friends(5);
      double v1[2] = {0., 1.};
      double v2[2];

      node = node_list[i];
      for (j=0; j<5; j++)
      {
        node_friend = node->friends_list[j];
        node_friend->getMapCoords(*node, v2);

        dot_prod = v1[0]*v2[0] + v1[1]*v2[1];
        mag1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1]);
        mag2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1]);

        inner_angle = acos(dot_prod/(mag1*mag2))*180./pi;
        det = v1[0]*v2[1] - v1[1]*v2[0];

        if (det < 0.0) ordered_friends[j].ang = inner_angle;
        else ordered_friends[j].ang = 360. - inner_angle;

        ordered_friends[j].ang = angles[j];
        ordered_friends[j].node = node_friend;
      }

      std::sort(ordered_friends.begin(), ordered_friends.end());

      for (j=0; j<5; j++)
      {
        node->friends_list[j] = ordered_friends[j].node;
      }

    }

    // for (i=0; i<node_list.size(); i++)
    // {
    //     node = node_list[i];
    //     node->printCoords();
    //
    //     std::cout<<node->ID<<'\t'<<std::endl;
    //     for (j=0; j<node->friend_num; j++)
    //     {
    //         // node->friends_list[j] = node->temp_friends[j];
    //         // node->updated[j] = 0;
    //         std::cout<<node->friends_list[j]->ID<<' ';
    //     }
    //     std::cout<<std::endl<<std::endl;
    //
    //     node->temp_friends.clear();
    // }
};

void Grid::bisectEdges(void)
{
    int node_count, i, j, k, added;
    double min_dist = 1.0;
    double dist;
    Node * node, *node_friend, *node_friend2;
    Node * inter_friend1, * inter_friend2;
    std::vector< Node * > new_nodes;

    node_count = this->node_list.size()-1;

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

    double dot_prod;
    double det;
    double mag1, mag2;
    double cosx, rad;
    double inner_angle;

    for (i=0; i<new_nodes.size(); i++)
    {
      //Order the friends of the new guys
      // double v1[2], v2[2];

      std::vector<double> angles(6);
      std::vector<AngleSort> ordered_friends(6);
      double v1[2] = {0., 1.};
      double v2[2];

      node = new_nodes[i];
      for (j=0; j<6; j++)
      {
        node_friend = node->friends_list[j];
        node_friend->getMapCoords(*node, v2);

        dot_prod = v1[0]*v2[0] + v1[1]*v2[1];
        mag1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1]);
        mag2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1]);

        inner_angle = acos(dot_prod/(mag1*mag2))*180./pi;
        det = v1[0]*v2[1] - v1[1]*v2[0];

        if (det < 0.0) ordered_friends[j].ang = inner_angle;
        else ordered_friends[j].ang = 360. - inner_angle;

        ordered_friends[j].node = node_friend;
      }

      std::sort(ordered_friends.begin(), ordered_friends.end());

      for (j=0; j<6; j++)
      {
        node->friends_list[j] = ordered_friends[j].node;
      }

    }

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

    node_list.insert(node_list.end(),new_nodes.begin(),new_nodes.end());


    // WE COULD ORDER NEW NODES HERE. ONCE A NODE HAS IT'S FRIENDS
    // ORDERED IT SHOULD NEVER DO IT AGAIN.



    // for (i=0; i<node_list.size(); i++)
    // {
    //     node = node_list[i];
    //     node->printCoords();
    //
    //     std::cout<<node->ID<<'\t'<<std::endl;
    //     for (j=0; j<node->friend_num; j++)
    //     {
    //         // node->friends_list[j] = node->temp_friends[j];
    //         // node->updated[j] = 0;
    //         std::cout<<node->friends_list[j]->ID<<' ';
    //     }
    //     std::cout<<std::endl<<std::endl;
    //
    //     node->temp_friends.clear();
    // }

};
