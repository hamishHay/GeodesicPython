#include "Node.h"
#include "math_functions.h"

Node::Node(double x, double y, double z, int ID_num=0)
{
  xyz_coords[0] = x;
  xyz_coords[1] = y;
  xyz_coords[2] = z;
  ID = ID_num;

  cart2sph(xyz_coords, sph_coords);
}
