#include "Node.h"
#include "Grid.h"
#include "math_functions.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string>

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

                dist = (*nodeA - *nodeB).getMagnitude();    // TODO - does this create a new node?

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
        // std::cout<<node->friends_list.size()<<std::endl;
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
      if (node->boundary >= 0)
      {
        int friend_num = node->friend_num;
        for (j=0; j<node->friend_num; j++)
        {
          if (node->friends_list[j]->boundary < 0) {
              friend_num--;
              node->friends_list[j]->ID = -2;
          }
        }

        // if (node->ID == 12) std::cout<<node->ID<<' '<<friend_num<<std::endl;

        std::vector<AngleSort> ordered_friends(friend_num);
        for (j=0; j<friend_num; j++)
        {
          node_friend = node->friends_list[j];
          node_friend->getMapCoords(*node, v2);

          ordered_friends[j].ang = atan2(v2[0], v2[1])*180./pi;// - 90.0 + 360.0;

          ordered_friends[j].node = node_friend;
        }

        std::sort(ordered_friends.begin(), ordered_friends.end());

        for (j=0; j<friend_num; j++)
        {
          node->friends_list[j] = ordered_friends[j].node;
        }

      }
    }
    for (i=0; i<node_list.size(); i++)
    {
      node = node_list[i];
      if (node->boundary == 1)
      {
        node->orderFriends();
        // int friend_num = node->friend_num;
        // for (j=0; j<node->friend_num; j++)
        // {
        //   if (node->boundary == 1)
        //   {
        //     // node->boundary = 1;
        //     break;
        //   }
        // }

      }
    }
}

void Grid::twistGrid(void)
{
    int i, j, k, g;
    Node * node, *node_friend, *node_friend2;
    double c1[3] = {0.0, 0.0, 0.0};
    double max_dist = pi/180. * 30.;
    double rot_angle = pi/5.;
    double tol=1.2;

    std::vector< int > interior_nodes;

    double avg_dist = 0.0;
    for (i=0; i<node_list.size(); i++)
    {
        node = node_list[i];

        double node_avg_dist = 0.0;
        for (j=0; j<node->friend_num; j++)
        {
            node_avg_dist += sphericalLength(node->sph_coords, node->friends_list[j]->sph_coords);
        }

        avg_dist += node_avg_dist/node->friend_num;
    }
    avg_dist /= node_list.size();

    // std::cout<<avg_dist<<std::endl;

    std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
        node = node_list[i];

        if (node->xyz_coords[2] < 0.0) node->transformSph(rot_angle);


    }

    for (i=0; i<node_list.size(); i++)
    {

        std::vector<int> surrounding_nodes;

        Node * nodeA = this->node_list[i];

        if ( fabs(nodeA->sph_coords[1]) < avg_dist*tol )
        {
            std::vector<int> remove_index;
            for (j=0; j<nodeA->friend_num; j++)
            {
                Node * nodeB = nodeA->friends_list[j];
                // double dist2 = (*nodeA - *nodeB).getMagnitude();    // TODO - does
                double dist2 = sphericalLength(nodeA->sph_coords, nodeB->sph_coords);
                //this create a new node?

                // if (i==640) std::cout<<' '<<nodeA->ID<<' '<<nodeB->ID<<' '<<dist2<<' '<<avg_dist<<std::endl;

                if (dist2 > avg_dist*tol)
                {

                    // nodeA->friends_list[k] = nodeB;
                    // if (i==640) std::cout<<"REMOVE "<<nodeA->ID<<' '<<nodeB->ID<<' '<<dist2<<' '<<avg_dist<<std::endl;

                    remove_index.push_back(j);
                }
            }

            int count=0;
            for (j=0; j<remove_index.size(); ++j)
            {
                nodeA->friends_list.erase(nodeA->friends_list.begin() + remove_index[j]-count);
                nodeA->friend_num--;
                count++;
            }

            // std::cout<<nodeA->sph_coords[1]*180./pi<<std::endl;
            for (j=0; j<node_list.size(); j++)
            {
                Node * nodeB = this->node_list[j];
                if ( (i != j) && (fabs(nodeB->sph_coords[1]) < avg_dist*tol) )
                {

                    double dist = sphericalLength(nodeA->sph_coords, nodeB->sph_coords);    // TODO - does this create a new node?


                    if (dist < avg_dist*tol && !nodeA->isFriend(nodeB))
                    {

                        std::cout<<i<<' '<<j<<' '<<dist<<' '<<avg_dist<<std::endl;

                        std::cout<<"REPLACING "<<nodeA->ID<<' '<<nodeB->ID<<' '<<dist<<' '<<avg_dist<<' '<<nodeA->friend_num<<std::endl;

                        nodeA->printCoords();
                        nodeB->printCoords();

                        nodeA->addFriend(nodeB);
                        nodeA->friend_num++;


                    }
                }
            }
        }

        // surrounding_nodes.push_back(-1);
        // nodeA->friends_list.push_back(NULL);
        //
        // this->friends_list.push_back( surrounding_nodes );
    }

    // for (i=0; i<node_list.size(); i++)
    // {
    //     node = node_list[i];
    //
    //     if (  )
    //     {
    //
    //     }
    // }

}

void Grid::applyBoundary(void)
{
    int i, j, k;
    Node * node, *node_friend, *node_friend2;
    double c1[3] = {0.0, 0.0, 0.0};
    double dist = 0.0;
    double max_dist = pi/180. * 30.;

    std::vector< int > interior_nodes;

    // std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;
    for (i=0; i<node_list.size(); i++)
    {
        node = node_list[i];

        dist = sphericalLength(node->sph_coords, c1);

        if (dist <= max_dist){
            interior_nodes.push_back(node->ID);
            node->boundary = 0;
        }
        else
        {
            node->boundary = -1;
        }
    }

    for (i=0; i<interior_nodes.size(); i++)
    {
        node = node_list[ interior_nodes[i] ];
        // node->ID = i;

        for (j=0; j<node->friend_num; j++)
        {
            // if (node->friends_list[j]->ID == -2)
            // {
            //     node->boundary = 1;
            //     break;
            // }
            if (node->friends_list[j]->boundary == -1)
            {
                node->boundary = 1;
                break;
            }
        }

    }



}

void Grid::refineBoundary(void)
{
    int i, j, k;
    double mag;
    Node * node, * next_node, * last_node, * first_node;
    int bnum = 301;
    double boundary_angle = pi/180. * 30.;
    std::vector<Node *> boundary_nodes;
    std::vector<double> ang_diff;

    struct NextNode {
        Node * node_pointer;
        double ang;
    };

    // Find the angle between the boundary and the boundary node positions
    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
        // std::cout<<i<<' '<<node->ID<<' '<<node_list.size()<<std::endl;
        if (node->boundary == 1)
        {
            boundary_nodes.push_back(node);

            ang_diff.push_back( boundary_angle - acos(node->xyz_coords[0]) );
        }
    }

    // Get element index of the node closest to the boundary (smallest ang_diff)
    // Note that there are probably multiple minimums in this vector, but only
    // minimum is needed here.
    int indx_closest = std::distance(std::begin(ang_diff), std::min_element( std::begin(ang_diff), std::end(ang_diff) ) );
    ang_diff.clear();

    // Retrieve closest node
    node = boundary_nodes[indx_closest];

    // Purge boundary node vector and then add the closest node
    boundary_nodes.clear();
    boundary_nodes.push_back( node );

    // Now find the friend of node that is closest to the boundary, and propagate
    // along the boundary, repeating this. Note that the friend that is closest
    // to the boundary may be inside *or* outside the boundary

    last_node = node;
    next_node = node->friends_list[0];
    first_node = node;

    double v1[3] = {0., 0., 0.};
    double v2[3] = {0., 0., 0.};
    double vz[3] = {0., 0., 1.};

    v1[1] = -node->xyz_coords[1]; // <------ bad nasty hack :(
    v1[2] = node->xyz_coords[2];

    do
    {

        std::vector<NextNode> next_nodes;
        for (j=0; j<node->friend_num; j++)
        {
            NextNode potential_node;
            potential_node.node_pointer = node->friends_list[j];
            potential_node.ang = fabs(boundary_angle - acos(node->friends_list[j]->xyz_coords[0]));
            v2[1] = -potential_node.node_pointer->xyz_coords[1]; // <------ bad nasty hack :(
            v2[2] = potential_node.node_pointer->xyz_coords[2];

            // retrieve angle between z axis and the node's position projected
            // onto the z-y plane.
            double dot = dotProduct(v1, v2);
            double det = v1[1]*v2[2] - v1[2]*v2[1];
            double first_ang = atan2(det, dot);

            if (first_ang < 0.0) next_nodes.push_back(potential_node);

            // std::cout<<potential_node.node_pointer->ID<<' '<<potential_node.ang<<std::endl;
        }
        std::sort(next_nodes.begin(), next_nodes.end(),
               [](const auto& x, const auto& y) { return x.ang < y.ang; } );
       // for (j=0; j<next_nodes.size(); j++)
       // {
       //     // std::cout<<'\t'<<next_nodes[j].node_pointer->ID<<' '<<next_nodes[j].ang<<std::endl;
       // }

       next_node = next_nodes[0].node_pointer;
       if (next_nodes.size() > 1) {
           std::cout<<next_nodes[1].ang/next_nodes[0].ang<<' '<<next_nodes[1].node_pointer->boundary<<std::endl;
           if ((next_nodes[1].node_pointer->boundary == -1)
                && (next_nodes[1].ang < 1.8*next_nodes[0].ang))
           next_node = next_nodes[1].node_pointer;

       }

       v2[1] = -next_node->xyz_coords[1]; // <------ bad nasty hack :(
       v2[2] = next_node->xyz_coords[2];

       // retrieve angle between z axis and the node's position projected
       // onto the z-y plane.
       double dot = dotProduct(vz, v2);
       double det = vz[1]*v2[2] - vz[2]*v2[1];
       double offset = atan2(det, dot);

       double new_xyz[3];
       new_xyz[1] = sin(boundary_angle)*sin(offset);
       new_xyz[2] = sin(boundary_angle)*cos(offset);
       new_xyz[0] = sqrt(1. - pow(new_xyz[1], 2.0) - pow(new_xyz[2], 2.0));

       next_node->updateXYZ(new_xyz);

       next_nodes.clear();

       next_node->boundary = 2;

       boundary_nodes.push_back( next_node );
       last_node = node;
       node = next_node;

       v1[1] = -node->xyz_coords[1]; // <------ bad nasty hack :(
       v1[2] = node->xyz_coords[2];

       // for (j=0; j<boundary_nodes.size(); j++) std::cout<<boundary_nodes[j]->ID<<", ";
       // std::cout<<std::endl;
       // std::cout<<last_node->ID<<" is linked to "<<node->ID<<' '<<first_node->ID<<std::endl;
   }
   while (node != first_node);



   int count = 0;
   for (i=0; i<node_list.size(); i++)
   {
       node = this->node_list[i];
       if ((node->boundary == 0) || (node->boundary == 1)) node->boundary = 0;
       if (node->boundary == 2) node->boundary = 1;

       if (node->boundary >= 0) node->ID = count++;
   }

   this->orderFriends();

   count = 0;
   for (i=0; i<boundary_nodes.size(); i++)
   {
       node = boundary_nodes[i];
       int interior_friend_num = 0;
       int indx;
       for (j=0; j<node->friend_num; j++)
       {
           if (node->friends_list[j]->boundary == 0) {
               interior_friend_num++;
               indx = j;
           }
       }

       if ((interior_friend_num == 1)) {
           // std::cout<<node->ID<<std::endl;
           for (j=0; j<node->friend_num; j++)
           {
               std::cout<<node->friends_list[j]->ID<<' ';

           }
           std::cout<<indx<<std::endl;

           double cnode_sph[3], lnode_sph[3], rnode_sph[3], inode_sph[3];

           for (k=0; k<3; k++) {
             cnode_sph[k] = node->sph_coords[k];
             inode_sph[k] = node->friends_list[indx]->sph_coords[k];
             lnode_sph[k] = node->friends_list[indx-1]->sph_coords[k];
             rnode_sph[k] = node->friends_list[indx+1]->sph_coords[k];
           }
           // cnode_sph[1] = pi*0.5 - cnode_sph[1];
           // inode_sph[1] = pi*0.5 - inode_sph[1];
           // lnode_sph[1] = pi*0.5 - lnode_sph[1];
           // rnode_sph[1] = pi*0.5 - rnode_sph[1];


           double l_area, r_area;
           l_area = sphericalArea(cnode_sph, lnode_sph, inode_sph);
           r_area = sphericalArea(cnode_sph, inode_sph, rnode_sph);

           std::cout<<node->ID<<' '<<node->friends_list[indx]->ID<<' '<<node->friends_list[indx-1]->ID<<' '<<l_area<<std::endl;
           std::cout<<node->ID<<' '<<node->friends_list[indx]->ID<<' '<<node->friends_list[indx+1]->ID<<' '<<r_area<<std::endl;
           std::cout<<std::endl;
           // for (j=0; j<node->friend_num; j++)
           // {
           //     if (node->friends_list[j]->boundary == 0) {
           //         interior_friend_num++;
           //         indx = j;
           //     }
           // }

           // Node * replacement_node = node->friends_list[indx];
           //
           // double xyz[3];
           // xyz[0] = node->xyz_coords[0];
           // xyz[1] = node->xyz_coords[1];
           // xyz[2] = node->xyz_coords[2];
           //
           // replacement_node->updateXYZ(xyz);
           // replacement_node->ID = node->ID;
           // replacement_node->boundary = 1;
           //
           // // std::cout<<replacement_node->ID<<' '<<i<<std::endl;
           //
           // node->boundary = -1;
           // node->ID = -2;
           //
           // boundary_nodes[i] = replacement_node;
           // count++;
       }
   }
   //
   // std::cout<<count<<' '<<boundary_nodes.size()<<std::endl;
   //
   count = 0;
   for (i=0; i<node_list.size(); i++)
   {
       node = this->node_list[i];
       // if ((node->boundary == 0) || (node->boundary == 1)) node->boundary = 0;
       // if (node->boundary == 2) node->boundary = 1;

       if (node->boundary >= 0) node->ID = count++;
   }

   // std::cout<<std::endl;


}

void Grid::shiftNodes(void)
{
    int i, j, k;
    double mag;
    Node * node1, *node2, *node3;
    double sph1[3], sph2[3], sph3[3];
    double xy_sub_centers[6][3];
    double xy1[3], xy2[3], xy3[3];
    double xy_new_center[3];
    double new_xyz[3];
    double diff_xyz[k];
    double sph_center[3];
    double areas[6];
    double area;
    std::vector<std::vector<double>> shifted_xyz(node_list.size(), std::vector<double> (3));
    double residual = 1.0;
    double residual_old = 0.0;
    double e_converge = 1e-20;
    double r = 0.0;
    int iter;

    // while loop --> until converged

    // call find centroid function
    // for loop over every node
        // compute control volume areas
        // find the shifted node point
        // compare shifted point to old point to find residual
        // update old node
        // call find centroid function
    iter = 0;
    this->findCentroids();
    // std::cout<<"HI"<<std::endl;
    while ((iter < 2000))// && residual > e_converge)//(residual > e_converge)
    {
        residual_old = residual;
        residual = 0.0;
        for (i=0; i<node_list.size(); i++)
        {
            node1 = this->node_list[i];
            if (node1->boundary == 0)
            {
                // std::cout<<node1->ID<<' '<<node1->boundary<<std::endl;
                for (k=0; k<3; k++) {
                    sph1[k] = node1->sph_coords[k];    //latitude
                    xy1[k] = node1->xyz_coords[k];
                }

                xy_new_center[0] = 0.0;
                xy_new_center[1] = 0.0;
                xy_new_center[2] = 0.0;

                for (j=0; j<node1->friend_num; j++)
                {
                    for (k=0; k<3; k++)
                    {
                        sph2[k] = node1->centroids[j][k];
                        sph3[k] = node1->centroids[(j+1)%node1->friend_num][k];
                        // std::cout<<node1->ID<<' '<<sph3[k]*180./pi<<' '<<sph3[k]*180./pi<<std::endl;
                    }

                    areas[j] = sphericalArea(sph1, sph2, sph3);
                    // std::cout<<j<<'\t'<<areas[j]<<'\t'<<sph1[1]*180./pi<<'\t'<<sph2[1]*180./pi<<'\t'<<sph3[1]*180./pi<<std::endl;
                    // sph1[1] = pi*0.5 - sph1[1];
                    sph2[1] = pi*0.5 - sph2[1];
                    sph3[1] = pi*0.5 - sph3[1];

                    sph2cart(sph2, xy2);
                    sph2cart(sph3, xy3);

                    for (k=0; k<3; k++)
                        xy_new_center[k] += areas[j]*(xy1[k] + xy2[k] + xy3[k]);
                }

                mag = 0.0;
                for (k=0; k<3; k++) mag += xy_new_center[k]*xy_new_center[k];
                mag = sqrt(mag);

                r = 0.0;
                for (k=0; k<3; k++) {
                    xy_new_center[k] /= mag;

                    r += (xy_new_center[k] - xy1[k])*(xy_new_center[k] - xy1[k]);
                }
                r = sqrt(r);

                //if (node1->friend_num == 6)
                //{
                residual += r*r;
                node1 = this->node_list[i];
                node1->updateXYZ(xy_new_center);
                //}
            }
        }


        std::cout<<iter<<'\t'<<residual<<std::endl;

        this->findCentroids();

        iter += 1;


    }

}

void Grid::defineRegion(int reg, int subReg, int ID[])
{
    for (int i=0; i < 3; i++) regions[reg][subReg][i] = ID[i]; //this->node_list[ID[i]];
}

void Grid::findCentroids(void)
{
    int i, j, k;
    double mag;
    Node * node, *node_friend, *node_friend2;
    double xy_center[3], xy_center2[3];
    double sph_center[3];

    for (i=0; i<node_list.size(); i++)
    {
        node = this->node_list[i];
        if (node->boundary >= 0)
        {
            int skip_num = 0;
            if (node->dead_num) skip_num = node->dead_num + 1;
            for (j=0; j<node->friend_num - skip_num; j++)
            // for (j=0; j<node->friend_num; j++)
            {
                node_friend  = node->friends_list[j];
                node_friend2 = node->friends_list[(j+1)%node->friend_num];

                // if (node->ID ==3) std::cout<<node->ID<<' '<<node_friend->ID<<' '<<node_friend2->ID<<node->friend_num - node->dead_num<<std::endl;
                // for (k=0; k<3; k++) {
                //     xy_center[k] = (node->xyz_coords[k]
                //     + node_friend->xyz_coords[k]
                //     + node_friend2->xyz_coords[k]);
                // }
                //
                // mag = sqrt(xy_center[0]*xy_center[0] + xy_center[1]*xy_center[1] + xy_center[2]*xy_center[2]);
                // xy_center[0] /= mag;
                // xy_center[1] /= mag;
                // xy_center[2] /= mag;
                //
                // cart2sph(xy_center, sph_center);
                //
                // node->centroids[j][0] = sph_center[0];
                // node->centroids[j][1] = sph_center[1];
                // node->centroids[j][2] = sph_center[2];

                voronoiCenter(node->xyz_coords, node_friend->xyz_coords, node_friend2->xyz_coords, xy_center2);

                // std::cout<<xy_center[2]<<' '<<xy_center2[2]<<std::endl;

                cart2sph(xy_center2, sph_center);

                node->centroids[j][0] = sph_center[0];
                node->centroids[j][1] = sph_center[1];
                node->centroids[j][2] = sph_center[2];

                // if ((node_friend->boundary < 0) || (node_friend2->boundary < 0))
                // {
                //     node->centroids[j][0] = sph_center[0];
                //     node->centroids[j][1] = sph_center[1];
                //     node->centroids[j][2] = sph_center[2];
                //
                // }

            }

            if (node->friend_num == 5)
            {
                node->centroids[5][0] = -1.0;
                node->centroids[5][1] = -1.0;
                node->centroids[5][2] = -1.0;
            }
        }
    }
}

void Grid::orderNodesByRegion(void)
{
    int i, j, k, f, current_region, current_subregion, node_num, current_node_ID;
    int regIDs[3];
    double v1[3], v2[3], v3[3], v[3];
    Node * current_node;
    Node * p1, * p2, * p3, * p;
    bool inside, onEdge;
    std::vector<int> potential_IDs;
    std::vector<int> immediate_fs(pow(2, this->recursion_lvl) - 1);

    node_num = this->node_list.size();

    std::vector<int> ordered_IDs (node_num);
    std::vector<Node *> ordered_nodes (node_num);
    std::vector<int> not_updated_IDs (this->node_list.size() - 12);
    std::vector<int> updated_IDs;

    // Add pentagons to respective regions
    inside_region[0].push_back(0);
    inside_region[0].push_back(1);

    for (i=0; i < node_num - 12; i++)
    {
        not_updated_IDs[i] = i + 12;
    }

    int top_node_ID;
    int node_f_ID;
    int row_num;
    int f_num;
    int x, y;
    Node * top_node_reg, * top_node_row, * node_f, * previous_node;


    for (i=0; i<5; i++)
    {
      // Find the first neighbour to the pole on edge 1
      for (j = 0; j<5; j++)
      {
        node_f = node_list[0]->friends_list[j];

        p1 = node_list[ regions[i][0][0] ];
        p2 = node_list[ regions[i][0][1] ];

        for (k = 0; k < 3; k++)
        {
          v1[k] = p1->xyz_coords[k];
          v2[k] = p2->xyz_coords[k];
          v[k]  = node_f->xyz_coords[k];
        }

        if (isOnEdge(v1, v2, v))
        {
          top_node_reg = node_f;
          inside_region[i].push_back(node_f->ID);

          if (i==1) std::cout<<top_node_reg->xyz_coords[0]<<'\t'<<top_node_reg->xyz_coords[1]<<'\t'<<top_node_reg->xyz_coords[2]<<std::endl;
          break;
        }
      }

      // Loop through first row using anticlockwise friends until dlon vector changes sign
      top_node_row = top_node_reg;
      current_node = top_node_row;
      previous_node = current_node;

      double dlon = -1.0;
      int nrow = 0;
      while (nrow < pow(2, this->recursion_lvl) - 1)
      {
        p1 = node_list[ regions[i][0][0] ];
        p2 = node_list[ regions[i][0][1] ];
        p3 = node_list[ regions[i][0][2] ];
        for (k = 0; k < 3; k++)
        {
          v1[k] = p1->xyz_coords[k];
          v2[k] = p2->xyz_coords[k];
          v3[k] = p3->xyz_coords[k];
        }

        f_num = current_node->friend_num;

        for (j = 0; j < f_num; j++)
        {
          node_f = current_node->friends_list[j];
          for (k = 0; k < 3; k++) v[k]  = node_f->xyz_coords[k];

          if (isInsideSphericalTriangle(v1, v2, v3, v) && !isOnEdge(v3, v1, v))
          {
            inside_region[i].push_back(node_f->ID);
            immediate_fs[nrow] = j;
            current_node = node_f;
            nrow++;

            if (i==1) std::cout<<current_node->xyz_coords[0]<<'\t'<<current_node->xyz_coords[1]<<'\t'<<current_node->xyz_coords[2]<<std::endl;
            break;
          }
        }
      }



      Node * f_node2;
      int extra;
      bool added;

      for (y=0; y<(nrow+1)*2 - 1; y++)
      {
          for (x=0; x<nrow; x++)
          {
              extra = 0;
              if (i == 0) extra = 2;

              f_num = node_list[ inside_region[i][y*(nrow+1) + x + extra] ]->friend_num;
              current_node = node_list[ inside_region[i][y*(nrow+1) + x + extra] ]->friends_list[ immediate_fs[x] + f_num - 1];

              if (x == 0)
              {
                  added = false;
                  int count = f_num-1;
                  while (!added)
                  {
                      current_node = top_node_row;
                      node_f = current_node->friends_list[count];
                      for (k = 0; k < 3; k++) v[k]  = node_f->xyz_coords[k];

                      for (j=0; j < 4; j++)
                      {
                          p1 = node_list[ regions[i][j][0] ];
                          p2 = node_list[ regions[i][j][1] ];
                          p3 = node_list[ regions[i][j][2] ];
                          for (k = 0; k < 3; k++)
                          {
                              v1[k] = p1->xyz_coords[k];
                              v2[k] = p2->xyz_coords[k];
                              v3[k] = p3->xyz_coords[k];
                          }

                          if (isInsideSphericalTriangle(v1,v2,v3,v) &&
                              isOnEdge(v1, v2, v) &&
                              node_f->ID != inside_region[i][(y-1)*(nrow+1) + extra])
                          {
                              added = true;
                              current_node = node_f;
                              break;
                          }
                      }
                      count--;
                  }

                  top_node_row = current_node;
              }

              if (i==1) std::cout<<current_node->xyz_coords[0]<<'\t'<<current_node->xyz_coords[1]<<'\t'<<current_node->xyz_coords[2]<<std::endl;

              inside_region[i].push_back(current_node->ID);
          }

          f_node2 = node_list[ inside_region[i][y*(nrow+1) + x + extra] ];

          for (j=0; j<6; j++)
          {
              node_f = current_node->friends_list[j];

              if (current_node->friends_list[(j+1)%6] == f_node2)
              {
                  if (i==1) std::cout<<node_f->xyz_coords[0]<<'\t'<<node_f->xyz_coords[1]<<'\t'<<node_f->xyz_coords[2]<<std::endl;
                  inside_region[i].push_back(node_f->ID);
                  break;
              }

          }

      }

      // std::cout<<inside_region[i].size()-2<<", "<<nrow<<std::endl;

        // Return to beginning of row and find first clockwise friend in region

        // Find next clockwise friend that also is friends with node at ordered_list - nrow
    }

    int count = 0;
    for (i=0; i<5; i++)
    {
        for (j=0; j<inside_region[i].size(); j++)
        {
            ordered_IDs[count] = inside_region[i][j];
            count++;
        }
    }

    for (i=0; i<node_num; i++)
    {
        ordered_nodes[i] = node_list[ordered_IDs[i]];
        ordered_nodes[i]->ID = i;
    }

    for (i=0; i<node_num; i++)
    {
        node_list[i] = ordered_nodes[i];
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

    std::cout<<"SAVING"<<std::endl;

    std::string file_path = "grid_l"+std::to_string(recursion_lvl+1)+".txt";

    outFile = fopen(&file_path[0], "w");

    fprintf(outFile, "%-5s %-12s %-12s %-38s %-20s\n", "ID", "NODE LAT", "NODE LON", "FRIENDS LIST", "CENTROID COORD LIST");

    for (i=0; i<this->node_list.size(); i++)
    {
        node = this->node_list[i];

        if (node->boundary >= 0)
        {
            // std::cout<<node->ID<<std::endl;

            lat = node->sph_coords[1]*180./pi;
            lon = node->sph_coords[2]*180./pi + 180.0;

            if (lon>359.99) lon = 0.0;
            if (i<2) lon = 180.0;
            for (j=0; j<node->friend_num; j++)
            {
                f[j] = node->friends_list[j]->ID;

                if (f[j] >= 0)
                {
                    cx[j] = node->centroids[j][1]*180./pi;
                    cy[j] = node->centroids[j][2]*180./pi + 180.0;
                    if (cy[j]>359.99) cy[j] =0.0;
                }
                else
                {
                    cx[j] = -1.0;
                    cy[j] = -1.0;
                }
            }
            for (j=node->friend_num; j<6; j++)
            {
                // if (node->friend_num == 5) {
                    f[j] = -1;
                    cx[j] = -1.0;
                    cy[j] = -1.0;
                // }
            }
            fprintf(outFile, "%-5d %12.16f %12.16f { %4d, %4d, %4d, %4d, %4d, %4d}, {( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f)} \n",
            node->ID, lat, lon, f[0], f[1], f[2], f[3], f[4], f[5],
            cx[0], cy[0], cx[1], cy[1], cx[2], cy[2], cx[3], cy[3], cx[4], cy[4], cx[5], cy[5]);
        }

    }


    fclose(outFile);
}
