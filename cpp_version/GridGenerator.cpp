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

  sph[0] = r; sph[1] = 0.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n0(xyz, 0);
  n0.friend_num = 5;

  sph[0] = r; sph[1] = 180.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n1(xyz, 1);
  n1.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0;
  sph2cart(sph, xyz, false);
  Node n2(xyz, 2);
  n2.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*3.0;
  sph2cart(sph, xyz, false);
  Node n3(xyz, 3);
  n3.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*5.0;
  sph2cart(sph, xyz, false);
  Node n4(xyz, 4);
  n4.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*7.0;
  sph2cart(sph, xyz, false);
  Node n5(xyz, 5);
  n5.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*9.0;
  sph2cart(sph, xyz, false);
  Node n6(xyz, 6);
  n6.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  Node n7(xyz, 7);
  n7.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*2.0;
  sph2cart(sph, xyz, false);
  Node n8(xyz, 8);
  n8.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*4.0;
  sph2cart(sph, xyz, false);
  Node n9(xyz, 9);
  n9.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*6.0;
  sph2cart(sph, xyz, false);
  Node n10(xyz, 10);
  n10.friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*8.0;
  sph2cart(sph, xyz, false);
  Node n11(xyz, 11);
  n11.friend_num = 5;

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

  grid.findFriends();

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
  grid.findCentroids();
  grid.shiftNodes();

  std::cout<<std::endl<<"Grid generated. Total node #: "<<grid.node_list.size()<<std::endl;

  grid.saveGrid2File();

  return 1;
};
