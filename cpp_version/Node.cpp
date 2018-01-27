#include "Node.h"
#include "math_functions.h"

Node::Node(double xyz[], int ID_num)
{
  xyz_coords[0] = xyz[0];
  xyz_coords[1] = xyz[1];
  xyz_coords[2] = xyz[2];
  ID = ID_num;

  cart2sph(xyz_coords, sph_coords);
};
