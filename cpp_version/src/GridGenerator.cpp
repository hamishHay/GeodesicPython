//MAIN FILE

#include "Node.h"
#include "Grid.h"
#include "math_functions.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <stdexcept>

int main(int argc, char* argv[])
{
  Grid grid;

  double xyz[3];
  double sph[3];
  double r = 1.0;

  // Spherical coords here defined in colatitude!!!
  sph[0] = r; sph[1] = 0.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n0(xyz, 0);
  n0.friend_num = 5;
  n0.pentagon = 1;

  sph[0] = r; sph[1] = 180.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n1(xyz, 1);
  n1.friend_num = 5;
  n1.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0;
  sph2cart(sph, xyz, false);
  Node n2(xyz, 2);
  n2.friend_num = 5;
  n2.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*3.0;
  sph2cart(sph, xyz, false);
  Node n3(xyz, 3);
  n3.friend_num = 5;
  n3.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*5.0;
  sph2cart(sph, xyz, false);
  Node n4(xyz, 4);
  n4.friend_num = 5;
  n4.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*7.0;
  sph2cart(sph, xyz, false);
  Node n5(xyz, 5);
  n5.friend_num = 5;
  n5.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*9.0;
  sph2cart(sph, xyz, false);
  Node n6(xyz, 6);
  n6.friend_num = 5;
  n6.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n7(xyz, 7);
  n7.friend_num = 5;
  n7.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*2.0;
  sph2cart(sph, xyz, false);
  Node n8(xyz, 8);
  n8.friend_num = 5;
  n8.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*4.0;
  sph2cart(sph, xyz, false);
  Node n9(xyz, 9);
  n9.friend_num = 5;
  n9.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*6.0;
  sph2cart(sph, xyz, false);
  Node n10(xyz, 10);
  n10.friend_num = 5;
  n10.pentagon = 1;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*8.0;
  sph2cart(sph, xyz, false);
  Node n11(xyz, 11);
  n11.friend_num = 5;
  n11.pentagon = 1;

  int subregion0[3]  = {0, 2, 6};
  int subregion1[3]  = {2, 7, 6};
  int subregion2[3]  = {2, 8, 7};
  int subregion3[3]  = {8, 1, 7};
  int subregion4[3]  = {0, 3, 2};
  int subregion5[3]  = {3, 8, 2};
  int subregion6[3]  = {3, 9, 8};
  int subregion7[3]  = {9, 1, 8};
  int subregion8[3]  = {0, 4, 3};
  int subregion9[3]  = {4, 9, 3};
  int subregion10[3] = {4, 10, 9};
  int subregion11[3] = {10, 1, 9};
  int subregion12[3] = {0, 5, 4};
  int subregion13[3] = {5, 10, 4};
  int subregion14[3] = {5, 11, 10};
  int subregion15[3] = {11, 1, 10};
  int subregion16[3] = {0, 6, 5};
  int subregion17[3] = {6, 11, 5};
  int subregion18[3] = {6, 7, 11};
  int subregion19[3] = {7, 1, 11};

  grid = Grid();

  grid.addNode(&n0);
  grid.addNode(&n1);
  grid.addNode(&n2);
  grid.addNode(&n3);

  grid.addNode(&n4);
  grid.addNode(&n5);
  grid.addNode(&n6);
  grid.addNode(&n7);

  grid.addNode(&n8);
  grid.addNode(&n9);
  grid.addNode(&n10);
  grid.addNode(&n11);

  grid.defineRegion(0, 0, subregion0);
  grid.defineRegion(0, 1, subregion1);
  grid.defineRegion(0, 2, subregion2);
  grid.defineRegion(0, 3, subregion3);

  grid.defineRegion(1, 0, subregion4);
  grid.defineRegion(1, 1, subregion5);
  grid.defineRegion(1, 2, subregion6);
  grid.defineRegion(1, 3, subregion7);

  grid.defineRegion(2, 0, subregion8);
  grid.defineRegion(2, 1, subregion9);
  grid.defineRegion(2, 2, subregion10);
  grid.defineRegion(2, 3, subregion11);

  grid.defineRegion(3, 0, subregion12);
  grid.defineRegion(3, 1, subregion13);
  grid.defineRegion(3, 2, subregion14);
  grid.defineRegion(3, 3, subregion15);

  grid.defineRegion(4, 0, subregion16);
  grid.defineRegion(4, 1, subregion17);
  grid.defineRegion(4, 2, subregion18);
  grid.defineRegion(4, 3, subregion19);

  grid.findFriends();
  grid.orderFriends();
  // grid.twistGrid();

  std::istringstream iss( argv[1] );
  int N;

  if (iss >> N) {}
  else throw std::invalid_argument( "Unsuccessful grid recursion argument! First argument must be an integer." );

  if (N < 0) throw std::invalid_argument( "Grid recursion level must be greater than zero." );

  for (int i=0; i<N; i++)
  {
    std::cout<<std::endl<<"Bisecting edges: recursion level "<< i+1 <<'.'<<std::endl;
    grid.bisectEdges();
  }
 
  grid.orderFriends();
  grid.twistGrid();
  grid.orderFriends();
  grid.findCentroids();
  grid.shiftNodes();

  grid.createVertices();
  grid.createFaces();

//   for (int i=0; i<grid.node_list.size(); i++)
//   {
//     Node * node = grid.node_list[i];
//     std::cout<<i<<std::endl;
//     for (int j=0; j<node->face_list.size(); j++)
//     {
//         std::cout<<' '<<node->face_list[j]->length;
//     }
//     std::cout<<std::endl;  
//   }

  grid.calculateProperties();

//   for (int i=0; i<grid.node_list.size(); i++)
//   {
//     Node * node = grid.node_list[i];
//     std::cout<<i<<' '<<node->area<<std::endl;
//     // for (int j=0; j<node->friends_list.size(); j++)
//     // {
//     //     std::cout<<' '<<node->friends_list[j]->ID<<' '<<node->node_dists[j]<<std::endl;;
//     // }
//     // std::cout<<std::endl;  
//   }

    // for (int i=0; i<grid.node_list.size(); i++)
    // {
    //     Node * node = grid.node_list[i];
    //     std::cout<<i<<std::endl;
    //     for (int j=0; j<node->face_list.size(); j++)
    //     {
    //         std::cout<<' '<<node->face_list[j]->ID<<' '<<node->face_list[j]->sph_normal[0]<<' '<<node->face_list[j]->sph_normal[1]<<' '<<node->face_dirs[j]<<std::endl;;
    //     }
    //     std::cout<<std::endl;  
    // }

    // for (int i=0; i<grid.face_list.size(); i++)
    // {
    //     Face * face = grid.face_list[i];
    //     std::cout<<face->ID<<' '<<face->sph_normal[0]<<' '<<face->sph_normal[1]<<std::endl;
    // }


//   std::cout<<std::endl<<"Grid generated. Total node #: "<<grid.node_list.size()<<std::endl;

  grid.saveGrid2File();
  grid.saveGrid2HDF5();

  // To do:
  // Normal vec direction 
  // Tangential vec direction
  // Face friends 
  // -->Weights for coriolis?
  
  // Should I also store distances?
  // Areas? etc? Or should that be for ODIS to calculate?

  // To do for MPI:
  // Split domain into parts --> this should probably be a post-process.
  // Can ODIS successfully read in each part and print out all grid info 
  // for the non-split case?


  return 1;
};
