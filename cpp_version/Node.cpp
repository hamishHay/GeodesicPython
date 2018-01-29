#include "Node.h"
#include "math_functions.h"
#include <iostream>

Node::Node(double xyz[], int ID_num)
{
  xyz_coords[0] = xyz[0];
  xyz_coords[1] = xyz[1];
  xyz_coords[2] = xyz[2];
  ID = ID_num;

  cart2sph(xyz_coords, sph_coords);

  // friends_list
};

Node::Node(const Node &other_node)
{
  this->xyz_coords[0] = other_node.xyz_coords[0];
  this->xyz_coords[1] = other_node.xyz_coords[1];
  this->xyz_coords[2] = other_node.xyz_coords[2];

  this->ID = other_node.ID;

  cart2sph(xyz_coords, sph_coords);
};

double Node::getMagnitude()
{
    double mag;

    mag = sqrt(this->xyz_coords[0]*this->xyz_coords[0]
              +this->xyz_coords[1]*this->xyz_coords[1]
              +this->xyz_coords[2]*this->xyz_coords[2]);

    return mag;
}

Node Node::operator+(const Node &other_node)
{
  double new_coords[3];

  new_coords[0] = this->xyz_coords[0] + other_node.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] + other_node.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] + other_node.xyz_coords[2];

  Node new_node(new_coords);

  return new_node;
}

Node Node::operator-(const Node &other_node)
{
  double new_coords[3];

  new_coords[0] = this->xyz_coords[0] - other_node.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] - other_node.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] - other_node.xyz_coords[2];

  Node new_node(new_coords);

  return new_node;
}

Node & Node::operator=(const Node &other_node)
{
  this->xyz_coords[0] = other_node.xyz_coords[0];
  this->xyz_coords[1] = other_node.xyz_coords[1];
  this->xyz_coords[2] = other_node.xyz_coords[2];

  // Maybe we shouldn't do this here?
  cart2sph(xyz_coords, sph_coords);

  return *this;
}
