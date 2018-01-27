//MAIN FILE

#include "Node.h"
#include "Grid.h"
#include "math_functions.h"
#include <math.h>
#include <iostream>

int main()
{
  Grid grid;

  Node *n0, *n1, *n2, *n3, *n4, *n5, *n6;
  Node *n7, *n8, *n9, *n10, *n11;

  double xyz[3];
  double sph[3];
  double r = 1.0;

  sph[0] = r; sph[1] = 0.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  n0 = new Node(xyz, 0);
  n0->friend_num = 5;

  sph[0] = r; sph[1] = 180.0; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  n1 = new Node(xyz, 1);
  n1->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0;
  sph2cart(sph, xyz, false);
  n2 = new Node(xyz, 2);
  n2->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*3.0;
  sph2cart(sph, xyz, false);
  n3 = new Node(xyz, 3);
  n3->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*5.0;
  sph2cart(sph, xyz, false);
  n4 = new Node(xyz, 4);
  n4->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*7.0;
  sph2cart(sph, xyz, false);
  n5 = new Node(xyz, 5);
  n5->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 - atan(0.5)*180./pi; sph[2] = 36.0*9.0;
  sph2cart(sph, xyz, false);
  n6 = new Node(xyz, 6);
  n6->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 0.0;
  sph2cart(sph, xyz, false);
  n7 = new Node(xyz, 7);
  n7->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*2.0;
  sph2cart(sph, xyz, false);
  n8 = new Node(xyz, 8);
  n8->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*4.0;
  sph2cart(sph, xyz, false);
  n9 = new Node(xyz, 9);
  n9->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*6.0;
  sph2cart(sph, xyz, false);
  n10 = new Node(xyz, 10);
  n10->friend_num = 5;

  sph[0] = r; sph[1] = 90.0 + atan(0.5)*180./pi; sph[2] = 36.0*8.0;
  sph2cart(sph, xyz, false);
  n11 = new Node(xyz, 11);
  n11->friend_num = 5;

  grid = Grid();

  grid.addNode(*n0);
  grid.addNode(*n1);
  grid.addNode(*n2);
  grid.addNode(*n3);

  grid.addNode(*n4);
  grid.addNode(*n5);
  grid.addNode(*n6);
  grid.addNode(*n7);

  grid.addNode(*n8);
  grid.addNode(*n9);
  grid.addNode(*n10);
  grid.addNode(*n11);

  return 1;
};
