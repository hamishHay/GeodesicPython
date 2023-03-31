#include "Node.h"
#include "Face.h"
#include "Vertex.h"
#include "Grid.h"
#include "math_functions.h"
#include "H5Cpp.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <cstring>
#include <filesystem>

template<typename T>
void Grid::saveToHDF5Group(hid_t * group, h5DataArray<T> * data, char const *name)
{
    int RANK = 1;
    hsize_t DIMS[1] = {data->size};
    hsize_t start[1] = {0};
    hsize_t end[1] = {data->size};
    hid_t data_space = H5Screate_simple(RANK, DIMS, NULL);
    hid_t data_set = H5Dcreate(*group, name, data->h5type, data_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t mem_space = H5Screate_simple(RANK, DIMS, NULL);

    H5Sselect_hyperslab(data_space, H5S_SELECT_SET, start, NULL, end, NULL);
    H5Dwrite(data_set, data->h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->data);

    H5Dclose(data_set);
    H5Sclose(mem_space);
    H5Sclose(data_space);
};

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
        // nodeA->friends_list.push_back(NULL); // WHY DID I DO THIS

        this->friends_list.push_back( surrounding_nodes );
    }

    double dot_prod;
    double det;
    double mag1, mag2;
    double cosx, rad;
    double inner_angle;

    // std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;

    // orderFriends();

    // for (i=0; i<node_list.size(); i++)
    // {
    //   //Order the friends of the new guys
    //   // double v1[2], v2[2];

    //   std::vector<AngleSort> ordered_friends(5);
    //   double v1[2] = {0., 1.};
    //   double v2[2];
    //   double angle;

    //   node = node_list[i];
    //   for (j=0; j<5; j++)
    //   {
    //     // std::cout<<node->friends_list.size()<<std::endl;
    //     node_friend = node->friends_list[j];
    //     node_friend->getMapCoords(*node, v2);

    //     dot_prod = v1[0]*v2[0] + v1[1]*v2[1];

    //     det = v1[0]*v2[1] - v1[1]*v2[0];

    //     ordered_friends[j].ang = atan2(dot_prod, det)*180./pi;

    //     ordered_friends[j].node = node_friend;
    //   }

    //   std::sort(ordered_friends.begin(), ordered_friends.end());

    //   for (j=0; j<5; j++)
    //   {
    //     node->friends_list[j] = ordered_friends[j].node;
    //   }

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
    std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;

    for (unsigned i=0; i<node_list.size(); i++) node_list[i]->orderFriendList();

    std::cout<<" complete!"<<std::endl;
};


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
        for (j=0; j<node->friends_list.size(); j++)
        {
            node_avg_dist += sphericalLength(node->sph_coords, node->friends_list[j]->sph_coords);
        }

        avg_dist += node_avg_dist/node->friend_num;
    }
    avg_dist /= node_list.size();

    // std::cout<<avg_dist<<std::endl;

    // std::cout<<'\t'<<"Ordering friends of new nodes..."<<std::flush;
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
            for (j=0; j<nodeA->friends_list.size(); j++)
            {
                Node * nodeB = nodeA->friends_list[j];
                // double dist2 = (*nodeA - *nodeB).getMagnitude();    // TODO - does
                double dist2 = sphericalLength(nodeA->sph_coords, nodeB->sph_coords);
                //this create a new node?

                if (dist2 > avg_dist*tol)
                {
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

                        // std::cout<<i<<' '<<j<<' '<<dist<<' '<<avg_dist<<std::endl;

                        // std::cout<<"REPLACING "<<nodeA->ID<<' '<<nodeB->ID<<' '<<dist<<' '<<avg_dist<<' '<<nodeA->friend_num<<std::endl;

                        // nodeA->printCoords();
                        // nodeB->printCoords();

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

};


/*
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



};
*/

/*
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

//    this->orderFriends();

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


};
*/

void Grid::shiftNodes(void)
{
    double mag_r;
    Node * node1;
    double sph1[3], sph2[3], sph3[3];
    double xy1[3], xy2[3], xy3[3];
    double xy_new_center[3];
    double areas[6];
    std::vector<std::vector<double>> shifted_xyz(node_list.size(), std::vector<double> (3));
    double residual = 1.0;
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
    findCentroids();
    std::cout<<std::endl<<std::endl;
    while (residual > 1e-12)// && residual > e_converge)//(residual > e_converge)
    {
        // residual_old = residual;
        residual = 0.0;
        for (unsigned i=0; i<node_list.size(); i++)
        {
            node1 = this->node_list[i];
            if (node1->boundary == 0)
            {
                // std::cout<<node1->ID<<' '<<node1->boundary<<std::endl;
                for (unsigned k=0; k<3; k++) {
                    sph1[k] = node1->sph_coords[k];    //latitude
                    xy1[k] = node1->xyz_coords[k];
                }

                xy_new_center[0] = 0.0;
                xy_new_center[1] = 0.0;
                xy_new_center[2] = 0.0;

                for (unsigned j=0; j<node1->friends_list.size(); j++)
                {
                    for (unsigned k=0; k<3; k++)
                    {
                        sph2[k] = node1->centroids[j][k];
                        sph3[k] = node1->centroids[(j+1)%node1->friends_list.size()][k];
                        // std::cout<<node1->ID<<' '<<sph3[k]*180./pi<<' '<<sph3[k]*180./pi<<std::endl;
                    }

                    areas[j] = sphericalArea(sph1, sph2, sph3);

                    // Convert to colat
                    sph2[1] = pi*0.5 - sph2[1];
                    sph3[1] = pi*0.5 - sph3[1];

                    sph2cart(sph2, xy2);
                    sph2cart(sph3, xy3);

                    for (unsigned k=0; k<3; k++)
                        xy_new_center[k] += areas[j]*(xy1[k] + xy2[k] + xy3[k]);
                }

                mag_r = 0.0;
                for (unsigned k=0; k<3; k++) mag_r += xy_new_center[k]*xy_new_center[k];
                mag_r = 1.0/sqrt(mag_r);

                r = 0.0;
                for (unsigned k=0; k<3; k++) {
                    xy_new_center[k] *= mag_r;

                    r += (xy_new_center[k] - xy1[k])*(xy_new_center[k] - xy1[k]);
                }
                // r = sqrt(r);

                residual += r;//*r;
                // node1 = this->node_list[i];
                node1->updateXYZ(xy_new_center);
            }
        }


        std::cout<<'\r'<<iter<<'\t'<<residual<<"   ";

        this->findCentroids();

        iter += 1;
    }

    std::cout<<std::endl;

};


template <typename T>
bool isInList(T * element, std::vector<T *> &list) {
    for (unsigned i=0; i<list.size(); i++) {

        // std::cout<<element->ID<<' '<<list[i]->ID<<std::endl;
        if (element == list[i]) {
            // std::cout<<std::endl;
            return true;
        }
    }
    return false;       
    
}

// Function to loop through every element and assign a region to it
void Grid::allocateElementsToRegions(void)
{
    Node * node;
    Face * face;
    Vertex * vertex;

    std::vector<Node *> list1_node;
    std::vector<Node *> list2_node;
    std::vector<Face *> list1_face;
    std::vector<Face *> list2_face;
    std::vector<Vertex *> list1_vertex;
    std::vector<Vertex *> list2_vertex;

    // Which regions neighbour each other
    // I.e. region 0 neigbours regions region_adj[0][:]
    std::vector< std::vector<int> > region_adj = { {1}, {0} }; 

    // region_node_list.push_back( list1_node );
    // region_node_list.push_back( list2_node );

    // region_face_list.push_back( list1_face );
    // region_face_list.push_back( list2_face );
    
    // region_vertex_list.push_back( list1_vertex );
    // region_vertex_list.push_back( list2_vertex );

    for (unsigned i=0; i<node_list.size(); i++) {
        node = node_list[i];

        // Find which region the node belongs to

        // Basic splitting between north and south hemisphere
        
        if (node->sph_coords[1] >= 0.0) {   // North of or at equator
            node->region = 0;
            list1_node.push_back(node);
        }
        else    {                            // South of equator
            node->region = 1;
            list2_node.push_back(node);    
        }

        
    }
    region_node_list.push_back( list1_node );
    region_node_list.push_back( list2_node );

    for (unsigned i=0; i<face_list.size(); i++) {
        face = face_list[i];

        if (face->sph_coords[1] >= 0.0) {   // North of or at equator
            face->region = 0;
            list1_face.push_back(face);
        }
        else    {                            // South of equator
            face->region = 1;
            list2_face.push_back(face);    
        }

        // std::cout<<face->sph_coords[1]<<' '<<face->n1->sph_coords[1]<<' '<<face->n2->sph_coords[1]<<' '<<face->v1->sph_coords[1]<<' '<<face->v2->sph_coords[1]<<std::endl;
    }
    region_face_list.push_back( list1_face );
    region_face_list.push_back( list2_face );

    


    for (unsigned i=0; i<vertex_list.size(); i++) {
        vertex = vertex_list[i];

        if (vertex->sph_coords[1] >= 0.0) {   // North of or at equator
            vertex->region = 0;
            list1_vertex.push_back(vertex);
        }
        else    {                            // South of equator
            vertex->region = 1;
            list2_vertex.push_back(vertex);    
        }
    }
    region_vertex_list.push_back( list1_vertex );
    region_vertex_list.push_back( list2_vertex );

    for (unsigned k=0; k<region_node_list.size(); k++) {
        std::vector<Node *> current_node_list = region_node_list[k];

        // Each node needs to find if it neighbours any ghosts
        for (unsigned i=0; i<current_node_list.size(); i++) {
            Node * node = current_node_list[i];

            node->RID = i;          // Give the node a local ID
            node->updateGhosts();

            // if (node->node_ghost_list.size() != 0) std::cout<<"Node "<<node->ID<<" has ghosts";
            // for (unsigned j=0; j<node->node_ghost_list.size(); j++)
            //      std::cout<<' '<<node->node_ghost_list[j]->ID;
            // if (node->node_ghost_list.size() != 0) std::cout<<std::endl;
        }
    }
    
    for (unsigned k=0; k<region_face_list.size(); k++) {
        std::vector<Face *> current_face_list = region_face_list[k];

        // Each face needs to find if it neighbours any ghosts
        for (unsigned i=0; i<current_face_list.size(); i++) {
            face = current_face_list[i];

            face->RID = i;          // Give the face a local ID
            face->updateGhosts();

            // if (face->node_ghost_list.size() != 0) std::cout<<"Face "<<face->ID<<" in region "<<k<<" has ghosts";
            // for (unsigned j=0; j<face->node_ghost_list.size(); j++)
            //      std::cout<<' '<<face->node_ghost_list[j]->ID<<" at lat "<<face->node_ghost_list[j]->sph_coords[1]<<", ";
            // if (face->node_ghost_list.size() != 0) std::cout<<std::endl;
        }
    }

    for (unsigned k=0; k<region_vertex_list.size(); k++) {
        std::vector<Vertex *> current_vertex_list = region_vertex_list[k];

        // Each vertex needs to find if it neighbours any ghosts
        for (unsigned i=0; i<current_vertex_list.size(); i++) {
            vertex = current_vertex_list[i];

            vertex->RID = i;        // Give the vertex a local ID
            vertex->updateGhosts();
        }
    }

    // for (unsigned k1=0; k1<region_adj.size(); k1++) {           // Region
    //     for (unsigned k2=0; k2<region_adj[k1].size(); k2++) {   // Neighouring region
    //         std::vector<Node *> current_node_ghost_list;
    //         std::vector<Node *> current_node_list = region_node_list[k2];

    //         Node * node;
    //         for (unsigned i=0; i<current_node_list.size(); i++) {
    //             node = current_node_list[i];
    //             if (node->region == k1) current_node_ghost_list.push_back(node);
    //         }
    //     }
    // }

    // Loop over each region
    for (unsigned k=0; k<region_node_list.size(); k++) {
        std::vector<Node *> current_node_list = region_node_list[k];
        std::vector<Face *> current_face_list = region_face_list[k];
        std::vector<Vertex *> current_vertex_list = region_vertex_list[k];
        std::vector<Node *> current_node_ghost_list;
        std::vector<Face *> current_face_ghost_list;
        std::vector<Vertex *> current_vertex_ghost_list;

        // Each node needs to find if it neighbours any ghosts
        for (unsigned i=0; i<current_node_list.size(); i++) {
            Node * node = current_node_list[i];
         
            // Add ghost nodes that are adjacent to nodes
            for (unsigned j=0; j<node->node_ghost_list.size(); j++) {
                Node * node_to_add = node->node_ghost_list[j];

                if (!isInList<Node>(node_to_add, current_node_ghost_list)) 
                    current_node_ghost_list.push_back(node_to_add);
            }

            for (unsigned j=0; j<node->face_ghost_list.size(); j++) {
                Face * face_to_add = node->face_ghost_list[j];

                if (!isInList<Face>(face_to_add, current_face_ghost_list)) 
                    current_face_ghost_list.push_back(face_to_add);
            }

            for (unsigned j=0; j<node->vertex_ghost_list.size(); j++) {
                Vertex * vertex_to_add = node->vertex_ghost_list[j];

                if (!isInList<Vertex>(vertex_to_add, current_vertex_ghost_list)) 
                    current_vertex_ghost_list.push_back(vertex_to_add);
            }
        }
        
        for (unsigned i=0; i<current_face_list.size(); i++) {
            Face * face = current_face_list[i];
            // std::cout<<k<<' '<<face->sph_coords[1]<<std::endl;
         
            Node * node_to_add;
            for (unsigned j=0; j<face->node_ghost_list.size(); j++)
            {
                // std::cout<<"   "<<face->node_ghost_list[j]->region<<std::endl;
                node_to_add = face->node_ghost_list[j];
                if (!isInList<Node>(node_to_add, current_node_ghost_list)) 
                    current_node_ghost_list.push_back(node_to_add);
            }

            for (unsigned j=0; j<face->face1_ghost_list.size(); j++) {
                Face * face_to_add = face->face1_ghost_list[j];

                if (!isInList<Face>(face_to_add, current_face_ghost_list)) 
                    current_face_ghost_list.push_back(face_to_add);
            }

            for (unsigned j=0; j<face->face2_ghost_list.size(); j++) {
                Face * face_to_add = face->face2_ghost_list[j];

                if (!isInList<Face>(face_to_add, current_face_ghost_list)) 
                    current_face_ghost_list.push_back(face_to_add);
            }

            for (unsigned j=0; j<face->vertex_ghost_list.size(); j++) {
                Vertex * vertex_to_add = face->vertex_ghost_list[j];

                if (!isInList<Vertex>(vertex_to_add, current_vertex_ghost_list)) 
                    current_vertex_ghost_list.push_back(vertex_to_add);
            }
        }

        for (unsigned i=0; i<current_vertex_list.size(); i++) {
            Vertex * vertex = current_vertex_list[i];
            // std::cout<<k<<' '<<vertex->sph_coords[1]<<std::endl;
         

            // for (unsigned j=0; j<vertex->vertex_ghost_list.size(); j++) {
            //     Vertex * vertex_to_add = vertex->vertex_ghost_list[j];

            //     if (!isInList<Vertex>(vertex_to_add, current_vertex_ghost_list)) 
            //         current_vertex_ghost_list.push_back(vertex_to_add);
            // }

            Node * node_to_add;
            for (unsigned j=0; j<vertex->node_ghost_list.size(); j++)
            {
                // std::cout<<"   "<<vertex->node_ghost_list[j]->region<<std::endl;
                node_to_add = vertex->node_ghost_list[j];
                if (!isInList<Node>(node_to_add, current_node_ghost_list)) 
                    current_node_ghost_list.push_back(node_to_add);
            }

            for (unsigned j=0; j<vertex->face_ghost_list.size(); j++) {
                Face * face_to_add = vertex->face_ghost_list[j];

                if (!isInList<Face>(face_to_add, current_face_ghost_list)) 
                    current_face_ghost_list.push_back(face_to_add);
            }
        }

        region_node_ghost_list.push_back( current_node_ghost_list );
        region_face_ghost_list.push_back( current_face_ghost_list );
    }

    // for (int k=0; k<region_node_ghost_list.size(); k++) {
    //     std::vector<Node *> current_list = region_node_ghost_list[k];
    //     for (unsigned i=0; i<current_list.size(); i++)
    //     {
    //         std::cout<<"Region "<<k<<" has ghost node "<<current_list[i]->ID<<std::endl;
    //     }
    // }

    // for (int k=0; k<region_face_ghost_list.size(); k++) {
    //     std::vector<Face *> current_list = region_face_ghost_list[k];
    //     for (unsigned i=0; i<current_list.size(); i++)
    //     {
    //         std::cout<<"Region "<<k<<" has ghost face "<<current_list[i]->ID<<std::endl;
    //     }
    // }

    std::cout<<region_node_list[0].size()<<' '<<region_node_list[1].size()<<std::endl;
    std::cout<<region_node_ghost_list[0].size()<<' '<<region_node_ghost_list[1].size()<<std::endl;

    std::cout<<std::endl;

    std::cout<<region_face_list[0].size()<<' '<<region_face_list[1].size()<<std::endl;
    std::cout<<region_face_ghost_list[0].size()<<' '<<region_face_ghost_list[1].size()<<std::endl;
    
}

// void Grid::defineRegion(int reg, int subReg, int ID[])
// {
//     for (int i=0; i < 3; i++) regions[reg][subReg][i] = ID[i]; //this->node_list[ID[i]];
// };

void Grid::findCentroids(void)
{
    double mag;
    double xy_center[3], xy_center2[3];
    double sph_coords[3];
    bool append;

    for (unsigned i=0; i<node_list.size(); i++)
    {
        Node * node = this->node_list[i];

        if (node->centroids.size() == 0) append = true;
        else append = false;

        for (unsigned j=0; j<node->friends_list.size(); j++)
        {
            Node * node_friend  = node->friends_list[j];
            Node * node_friend2 = node->friends_list[(j+1)%node->friends_list.size()];
            
            voronoiCenter(node->xyz_coords, node_friend->xyz_coords, node_friend2->xyz_coords, xy_center2);

            cart2sph(xy_center2, sph_coords);

            if (append) node->addCentroid(sph_coords);
            else node->updateCentroidPos(j, sph_coords);
        }
    }

};

/*
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

};
*/

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
            
            cx[5] = -1.0;
            cy[5] = -1.0;
            f[5]  = -1.0;
            for (j=0; j<node->friends_list.size(); j++)
            {
                f[j] = node->friends_list[j]->ID;

                cx[j] = node->centroids[j][1]*180./pi;
                cy[j] = node->centroids[j][2]*180./pi + 180.0;
                if (cy[j]>359.99) cy[j] =0.0;

            }

            fprintf(outFile, "%-5d %12.16f %12.16f { %4d, %4d, %4d, %4d, %4d, %4d}, {( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f), ( % 10.16f, % 10.16f)} \n",
            node->ID, lat, lon, f[0], f[1], f[2], f[3], f[4], f[5],
            cx[0], cy[0], cx[1], cy[1], cx[2], cy[2], cx[3], cy[3], cx[4], cy[4], cx[5], cy[5]);
        }

    }


    fclose(outFile);
};

// Function to create all unique vertexes that exist in the domain
// These are the same as the node centroids.
void Grid::createVertices(void)
{
    for (unsigned i=0; i<node_list.size(); i++) {
        Node * node1 = node_list[i];

        for (unsigned j=0; j<node1->friends_list.size(); j++) 
        {
            Node * node2 = node1->friends_list[j];
            Node * node3 = node1->friends_list[ (j+1)%node1->friends_list.size() ];

            bool shared = false;
            for (unsigned k=0; k<node1->vertex_list.size(); k++) {
                Vertex * vertex = node1->vertex_list[k];

                shared = vertex->sharedNode(node1, node2, node3);

                if (shared) break;
            }

            if (!shared){
                double xyz[3] = {1.0, 2.0, 3.0};

                voronoiCenter(node1->xyz_coords, node2->xyz_coords, node3->xyz_coords, xyz);

                Vertex * vertex = new Vertex(vertex_list.size(), xyz, node1, node2, node3);

                node1->vertex_list.push_back(vertex);
                node2->vertex_list.push_back(vertex);
                node3->vertex_list.push_back(vertex);

                vertex_list.push_back(vertex);
            }
        }

        // Sort order of vertexes. Important!
        node1->orderVertexList();
    }

    std::cout<<"GENERATED "<<vertex_list.size()<<" vertii."<<std::endl;
};

// Function to create all unique faces that exist in the domain
void Grid::createFaces(void)
{
    // int NODE_NUM = node_list.size();

    // Loop over all nodes, assigning a face with each neighbour
    // if one does not exist already.
    for (unsigned i=0; i<node_list.size(); i++)
    {
        Node * node1 = node_list[i];

        for (unsigned j=0; j<node1->friends_list.size(); j++) 
        {
            Node * node2 = node1->friends_list[j];

            // Does node1 already share a face with node2?
            bool shared = false;
            for (unsigned k=0; k<node1->face_list.size(); k++) {
                Face * face = node1->face_list[k];

                shared = face->sharedNode(node1, node2);

                if (shared) break;
            }

            // Node1 and node2 do not share an existing face, so create one!
            if (!shared) 
            {

                // First need to find common two vertices between node1 and node2.
                std::vector<Vertex *> vert (2);
                int count = 0;
                for (unsigned k1=0; k1<node1->vertex_list.size(); k1++)
                {
                    Vertex * v1 = node1->vertex_list[k1];

                    for (unsigned k2=0; k2<node2->vertex_list.size(); k2++) 
                    {

                        Vertex * v2 = node2->vertex_list[k2];
                        if (v1 == v2) {
                            vert[count] = v1;
                            count++;
                            break;
                        }
                    }

                    if (count == 2) break; // found the two common vertices
                }

                                        // FACE_ID       vertex1  vertex2  node1  node2
                Face * face = new Face(face_list.size(), vert[0], vert[1], node1, node2);

                face_list.push_back(face);

                node1->face_list.push_back(face);
                node2->face_list.push_back(face);
            }

        } 

    }

    // Order the storage order of faces for each node. Perhaps not necessary.
    for (unsigned i=0; i<node_list.size(); i++) node_list[i]->orderFaceList();

    std::cout<<"GENERATED "<<face_list.size()<<" faces."<<std::endl;
};

void Grid::calculateProperties(void)    
{
    for (unsigned i=0; i<face_list.size(); i++)
    {
        Face * face = face_list[i];

        face->updateCenterPos();        // Middle of face - Checked!
        face->updateLength();           // Arc length of face - Checked!
        face->updateNormalVec();        // Normal vector of face - Checked!
        face->updateFaceFriends();      // Faces shared by the nodes adjoining parent face
        face->updateIntersectPos();     // intersect of face with node-node vector - Checked! (small diff with face center, as expected)
        face->updateArea();             // face area with nodes and vertices
        // face->updateFaceFriends();
        
    }

    for (unsigned i=0; i<node_list.size(); i++)
    {
        Node * node = node_list[i];

        node->updateArea();             // Control volume area - Checked!
        node->updateNodeDists();        // Distance between node and its friends - Checked! 
        node->updateFaceDirs();         // If face normal is inwards or outwards - Checked!
    
        // TODO
        // node->updateVertexAreas();   // Should this be a vertex or node property? --> probably a node property
    }

    for (unsigned i=0; i<vertex_list.size(); i++)
    {
        Vertex * vertex = vertex_list[i];

        vertex->findFaces();
        vertex->updateFaceDirs();         // If the face normal circulates clockwise or anticlockwise
        vertex->updateArea();
        vertex->updateSubAreas();
    }

    // This update depends on the values of each vertex
    for (unsigned i=0; i<node_list.size(); i++)
    {
        Node * node = node_list[i];
        node->updateVertexAreas();        // Should this be a vertex or node property? --> probably a node property
    }

    // for (unsigned i=0; i<face_list.size(); i++)
    // {
    //     Face * face = face_list[i];

    //           // Faces shared by the nodes adjoining parent face 
    // }

};

void Grid::saveGrid2HDF5(void)
{
    char dataFile[1024];
    std::string file_name = "/grid_l"+std::to_string(recursion_lvl+1)+".h5";
    std::string file_path = std::filesystem::current_path().string() + file_name;
    std::strcpy(dataFile, file_path.c_str());
    
    unsigned FACE_NUM = face_list.size();
    unsigned NODE_NUM = node_list.size();
    unsigned VERTEX_NUM = vertex_list.size();

    hid_t file = H5Fcreate(dataFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);



    // SAVE FACE INFO -------------------------------------------------------------------
    hid_t face_group = H5Gcreate(file, "FACES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // std::cout<<H5::PredType::NATIVE_UINT<<std::endl;
    h5DataArray<unsigned> * face_ID = new h5DataArray<unsigned>(FACE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<FACE_NUM; i++) face_ID->data[i] = face_list[i]->ID;
    saveToHDF5Group(&face_group, face_ID, "ID");
    delete face_ID;

    h5DataArray<double> * face_length = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_length->data[i] = face_list[i]->length;
    saveToHDF5Group(&face_group, face_length, "LENGTH");
    delete face_length;

    h5DataArray<double> * face_lat = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_lat->data[i] = face_list[i]->sph_coords[1]*180.0/pi; // Convert to degrees
    saveToHDF5Group(&face_group, face_lat, "LAT");
    delete face_lat;

    h5DataArray<double> * face_lon = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_lon->data[i] = face_list[i]->sph_coords[2]*180.0/pi + 180.0;
    saveToHDF5Group(&face_group, face_lon, "LON");
    delete face_lon;

    h5DataArray<double> * intersect_lat = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) intersect_lat->data[i] = face_list[i]->sph_intersect[1]*180.0/pi; // Convert to degrees
    saveToHDF5Group(&face_group, intersect_lat, "INTERSECT_LAT");
    delete intersect_lat;

    h5DataArray<double> * intersect_lon = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) intersect_lon->data[i] = face_list[i]->sph_intersect[2]*180.0/pi + 180.0;
    saveToHDF5Group(&face_group, intersect_lon, "INTERSECT_LON");
    delete intersect_lon;

    h5DataArray<double> * intersect_length = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) intersect_length->data[i] = face_list[i]->length_intersect;
    saveToHDF5Group(&face_group, intersect_length, "INTERSECT_LENGTH");
    delete intersect_length;

    h5DataArray<double> * face_area = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_area->data[i] = face_list[i]->area;
    saveToHDF5Group(&face_group, face_area, "AREA");
    delete face_area;

    h5DataArray<double> * face_nx = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_nx->data[i] = face_list[i]->sph_normal[0];
    saveToHDF5Group(&face_group, face_nx, "NORMAL_VEC_LON");
    delete face_nx;

    h5DataArray<double> * face_ny = new h5DataArray<double>(FACE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<FACE_NUM; i++) face_ny->data[i] = face_list[i]->sph_normal[1];
    saveToHDF5Group(&face_group, face_ny, "NORMAL_VEC_LAT");
    delete face_ny;
    // -----------------------------------------------------------------------------------



    // SAVE NODE INFO -------------------------------------------------------------------
    hid_t node_group = H5Gcreate(file, "NODES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    h5DataArray<unsigned> * node_ID = new h5DataArray<unsigned>(NODE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<NODE_NUM; i++) node_ID->data[i] = node_list[i]->ID;
    saveToHDF5Group(&node_group, node_ID, "ID");
    delete node_ID;

    h5DataArray<double> * node_area = new h5DataArray<double>(NODE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<NODE_NUM; i++) node_area->data[i] = node_list[i]->area;
    saveToHDF5Group(&node_group, node_area, "AREA");
    delete node_area;

    h5DataArray<double> * node_lat = new h5DataArray<double>(NODE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<NODE_NUM; i++) node_lat->data[i] = node_list[i]->sph_coords[1]*180.0/pi; // Convert to degrees
    saveToHDF5Group(&node_group, node_lat, "LAT");
    delete node_lat;

    h5DataArray<double> * node_lon = new h5DataArray<double>(NODE_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<NODE_NUM; i++) node_lon->data[i] = node_list[i]->sph_coords[2]*180.0/pi + 180.0;
    saveToHDF5Group(&node_group, node_lon, "LON");
    delete node_lon;
    // -----------------------------------------------------------------------------------


    // SAVE VERTEX INFO -------------------------------------------------------------------
    hid_t vertex_group = H5Gcreate(file, "VERTICES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    h5DataArray<unsigned> * vertex_ID = new h5DataArray<unsigned>(VERTEX_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<VERTEX_NUM; i++) vertex_ID->data[i] = vertex_list[i]->ID;
    saveToHDF5Group(&vertex_group, vertex_ID, "ID");
    delete vertex_ID;

    h5DataArray<double> * vertex_lat = new h5DataArray<double>(VERTEX_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<VERTEX_NUM; i++) vertex_lat->data[i] = vertex_list[i]->sph_coords[1]*180.0/pi; // Convert to degrees
    saveToHDF5Group(&vertex_group, vertex_lat, "LAT");
    delete vertex_lat;

    h5DataArray<double> * vertex_lon = new h5DataArray<double>(VERTEX_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<VERTEX_NUM; i++) vertex_lon->data[i] = vertex_list[i]->sph_coords[2]*180.0/pi + 180.0;
    saveToHDF5Group(&vertex_group, vertex_lon, "LON");
    delete vertex_lon;

    h5DataArray<double> * vertex_area = new h5DataArray<double>(VERTEX_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<VERTEX_NUM; i++) vertex_area->data[i] = vertex_list[i]->area;
    saveToHDF5Group(&vertex_group, vertex_area, "AREA");
    delete vertex_area;

    // ---------------------------------------------------------------------------------------


    // INTERCONNECTIVITY ----------------------------------------------------------------------
    hid_t node_group_f = H5Gcreate(node_group, "FRIENDS", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // VERTEX --> NODES
    unsigned FRIENDS_NUM = 0;    
    for (unsigned i=0; i<NODE_NUM; i++) FRIENDS_NUM += node_list[i]->vertex_list.size();

    hid_t node_group_f_v = H5Gcreate(node_group_f, "VERTICES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * node_v_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    int count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->vertex_list.size(); j++) {
            node_v_ID->data[count] = node_list[i]->vertex_list[j]->ID;
            count++;
        }
    }
    saveToHDF5Group(&node_group_f_v, node_v_ID, "ID");
    delete node_v_ID;

    h5DataArray<unsigned> * node_v_fnum = new h5DataArray<unsigned>(NODE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<NODE_NUM; i++) node_v_fnum->data[i] = node_list[i]->vertex_list.size();
    saveToHDF5Group(&node_group_f_v, node_v_fnum, "FRIEND_NUM");
    delete node_v_fnum;

    h5DataArray<double> * node_v_area = new h5DataArray<double>(FRIENDS_NUM, H5T_NATIVE_DOUBLE);
    count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->vertex_list.size(); j++) {
            node_v_area->data[count] = node_list[i]->vertex_areas[j];
            count++; }}

    saveToHDF5Group(&node_group_f_v, node_v_area, "AREA");
    delete node_v_area;

    // FACES --> NODES ---------------------------------------------------------------------------
    FRIENDS_NUM = 0;
    for (unsigned i=0; i<NODE_NUM; i++) FRIENDS_NUM += node_list[i]->face_list.size();

    hid_t node_group_f_f = H5Gcreate(node_group_f, "FACES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * node_f_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->face_list.size(); j++) {
            node_f_ID->data[count] = node_list[i]->face_list[j]->ID;
            count++; }}

    saveToHDF5Group(&node_group_f_f, node_f_ID, "ID");
    delete node_f_ID;

    // FACE_DIR --> NODE
    h5DataArray<int> * node_f_dir = new h5DataArray<int>(FRIENDS_NUM, H5T_NATIVE_INT);
    count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->face_list.size(); j++) {
            node_f_dir->data[count] = node_list[i]->face_dirs[j];
            count++; }}

    saveToHDF5Group(&node_group_f_f, node_f_dir, "DIR");
    delete node_f_dir;

    h5DataArray<unsigned> * node_f_fnum = new h5DataArray<unsigned>(NODE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<NODE_NUM; i++) node_f_fnum->data[i] = node_list[i]->face_list.size();
    saveToHDF5Group(&node_group_f_f, node_f_fnum, "FRIEND_NUM");
    delete node_f_fnum;

    // NODES --> NODES ---------------------------------------------------------------------------
    FRIENDS_NUM = 0;
    for (unsigned i=0; i<NODE_NUM; i++) FRIENDS_NUM += node_list[i]->friends_list.size();

    hid_t node_group_f_n = H5Gcreate(node_group_f, "NODES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * node_n_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->friends_list.size(); j++) {
            node_n_ID->data[count] = node_list[i]->friends_list[j]->ID;
            count++; }}

    saveToHDF5Group(&node_group_f_n, node_n_ID, "ID");
    delete node_n_ID;

    // NODE-->NODE DISTANCE
    h5DataArray<double> * node_n_dist = new h5DataArray<double>(FRIENDS_NUM, H5T_NATIVE_DOUBLE);

    count=0;
    for (unsigned i=0; i<NODE_NUM; i++) {
        for (unsigned j=0; j<node_list[i]->friends_list.size(); j++) {
            node_n_dist->data[count] = node_list[i]->node_dists[j];
            count++; }}

    saveToHDF5Group(&node_group_f_n, node_n_dist, "DISTANCE");
    delete node_n_dist;

    h5DataArray<unsigned> * node_n_fnum = new h5DataArray<unsigned>(NODE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<NODE_NUM; i++) node_n_fnum->data[i] = node_list[i]->friends_list.size();
    saveToHDF5Group(&node_group_f_n, node_n_fnum, "FRIEND_NUM");
    delete node_n_fnum;



    
    // INTERCONNECTIVITY FOR FACES ----------------------------------------------------------------------
    hid_t face_group_f = H5Gcreate(face_group, "FRIENDS", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // NODE-->FACES
    FRIENDS_NUM = face_list.size()*2;  // Each face is next to only 2 nodes and 2 vertices

    hid_t face_group_f_n = H5Gcreate(face_group_f, "NODES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * face_n_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<FACE_NUM; i++) {
        face_n_ID->data[2*i]     = face_list[i]->n1->ID;
        face_n_ID->data[2*i + 1] = face_list[i]->n2->ID; }

    saveToHDF5Group(&face_group_f_n, face_n_ID, "ID");
    delete face_n_ID;

    // VERTEXES-->FACES
    hid_t face_group_f_v = H5Gcreate(face_group_f, "VERTICES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * face_v_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<FACE_NUM; i++) {
        face_v_ID->data[2*i]     = face_list[i]->v1->ID;
        face_v_ID->data[2*i + 1] = face_list[i]->v2->ID; }

    saveToHDF5Group(&face_group_f_v, face_v_ID, "ID");
    delete face_v_ID;

    // FACES-->FACES
    FRIENDS_NUM=0;
    for (unsigned i=0; i<FACE_NUM; i++) FRIENDS_NUM += face_list[i]->friends_list1.size();

    hid_t face_group_f_f1 = H5Gcreate(face_group_f, "FACES1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * face_f1_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    count = 0;
    for (unsigned i=0; i<FACE_NUM; i++) {
        for (unsigned j=0; j<face_list[i]->friends_list1.size(); j++) {
            face_f1_ID->data[count] = face_list[i]->friends_list1[j]->ID;
            count++; }}

    saveToHDF5Group(&face_group_f_f1, face_f1_ID, "ID");
    delete face_f1_ID;

    FRIENDS_NUM=0;
    for (unsigned i=0; i<FACE_NUM; i++) FRIENDS_NUM += face_list[i]->friends_list2.size();

    hid_t face_group_f_f2 = H5Gcreate(face_group_f, "FACES2", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * face_f2_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    count = 0;
    for (unsigned i=0; i<FACE_NUM; i++) {
        for (unsigned j=0; j<face_list[i]->friends_list2.size(); j++) {
            face_f2_ID->data[count] = face_list[i]->friends_list2[j]->ID;
            count++; }}

    saveToHDF5Group(&face_group_f_f2, face_f2_ID, "ID");
    delete face_f2_ID;

    // FACES_NUM-->FACES
    h5DataArray<unsigned> * face_f1_fnum = new h5DataArray<unsigned>(FACE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<FACE_NUM; i++) face_f1_fnum->data[i] = face_list[i]->friends_list1.size();
    
    saveToHDF5Group(&face_group_f_f1, face_f1_fnum, "FRIEND_NUM");
    delete face_f1_fnum;


    h5DataArray<unsigned> * face_f2_fnum = new h5DataArray<unsigned>(FACE_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<FACE_NUM; i++) face_f2_fnum->data[i] = face_list[i]->friends_list2.size();
    
    saveToHDF5Group(&face_group_f_f2, face_f2_fnum, "FRIEND_NUM");
    delete face_f2_fnum;


    // INTERCONNECTIVITY FOR VERTICES ----------------------------------------------------------------------
    hid_t vertex_group_f = H5Gcreate(vertex_group, "FRIENDS", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // NODE-->VERTICES
    FRIENDS_NUM = VERTEX_NUM*3;  // Each vertex is next to only 3 nodes and 3 vertices

    hid_t vertex_group_f_n = H5Gcreate(vertex_group_f, "NODES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * vertex_n_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<VERTEX_NUM; i++) {
        vertex_n_ID->data[3*i]     = vertex_list[i]->node_list[0]->ID;
        vertex_n_ID->data[3*i + 1] = vertex_list[i]->node_list[1]->ID;
        vertex_n_ID->data[3*i + 2] = vertex_list[i]->node_list[2]->ID; }
        
    saveToHDF5Group(&vertex_group_f_n, vertex_n_ID, "ID");
    delete vertex_n_ID;

    // NODE-->VERTEX AREAS
    h5DataArray<double> * vertex_n_area = new h5DataArray<double>(FRIENDS_NUM, H5T_NATIVE_DOUBLE);
    for (unsigned i=0; i<VERTEX_NUM; i++) {
        vertex_n_area->data[3*i]     = vertex_list[i]->subareas[0];
        vertex_n_area->data[3*i + 1] = vertex_list[i]->subareas[1];
        vertex_n_area->data[3*i + 2] = vertex_list[i]->subareas[2]; }
        
    saveToHDF5Group(&vertex_group_f_n, vertex_n_area, "SUBAREA");
    delete vertex_n_area;

    // FACES-->VERTICES
    hid_t vertex_group_f_f = H5Gcreate(vertex_group_f, "FACES", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5DataArray<unsigned> * vertex_f_ID = new h5DataArray<unsigned>(FRIENDS_NUM, H5T_NATIVE_UINT);
    for (unsigned i=0; i<VERTEX_NUM; i++) {
        vertex_f_ID->data[3*i]     = vertex_list[i]->face_list[0]->ID;
        vertex_f_ID->data[3*i + 1] = vertex_list[i]->face_list[1]->ID;
        vertex_f_ID->data[3*i + 2] = vertex_list[i]->face_list[2]->ID; }

    saveToHDF5Group(&vertex_group_f_f, vertex_f_ID, "ID");
    delete vertex_f_ID;

    // FACEDIR-->VERTICES
    h5DataArray<int> * vertex_f_dir = new h5DataArray<int>(FRIENDS_NUM, H5T_NATIVE_INT);
    for (unsigned i=0; i<VERTEX_NUM; i++) {
        vertex_f_dir->data[3*i]     = vertex_list[i]->face_dirs[0];
        vertex_f_dir->data[3*i + 1] = vertex_list[i]->face_dirs[1];
        vertex_f_dir->data[3*i + 2] = vertex_list[i]->face_dirs[2]; }

    saveToHDF5Group(&vertex_group_f_f, vertex_f_dir, "DIR");
    delete vertex_f_dir;
    
} 