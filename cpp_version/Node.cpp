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

  this->sph_coords[0] = other_node.sph_coords[0];
  this->sph_coords[1] = other_node.sph_coords[1];
  this->sph_coords[2] = other_node.sph_coords[2];

  this->ID = other_node.ID;
  this->friend_num = other_node.friend_num;

};

double Node::getMagnitude()
{
    double mag;

    mag = sqrt(this->xyz_coords[0]*this->xyz_coords[0]
              +this->xyz_coords[1]*this->xyz_coords[1]
              +this->xyz_coords[2]*this->xyz_coords[2]);

    return mag;
}

void Node::project2Sphere(double r)
{
    double mag;

    mag = this->getMagnitude();

    this->xyz_coords[0] /= mag;
    this->xyz_coords[1] /= mag;
    this->xyz_coords[2] /= mag;

    // Maybe we shouldn't do this here?
    cart2sph(xyz_coords, sph_coords);
}

void Node::addFriend(Node * n)
{
    friends_list.push_back(n);
}

void Node::addTempFriend(Node * n)
{
    temp_friends.push_back(n);
}

void Node::printCoords()
{
    // std::cout<<"Coordinates of node "<<this->ID<<": "<<std::endl;
    // std::cout<<"x: "<<this->xyz_coords[0]<<", ";
    // std::cout<<"y: "<<this->xyz_coords[1]<<", ";
    // std::cout<<"z: "<<this->xyz_coords[2]<<std::endl;

    std::cout<<"Coordinates of node "<<this->ID<<": "<<std::endl;
    std::cout<<"lat: "<<this->sph_coords[1]*180./pi<<", ";
    std::cout<<"lon: "<<this->sph_coords[2]*180./pi<<std::endl;
}

void Node::getMapCoords(const Node &center_node, double xy[])
{
  double m;
  double lat1, lat2, lon1, lon2;

  lat1 = center_node.sph_coords[1];
  lon1 = center_node.sph_coords[2];
  lat2 = this->sph_coords[1];
  lon2 = this->sph_coords[2];

  m = 2.0 / (1.0 + sin(lat2)*sin(lat1) + cos(lat1)*cos(lat2)*cos(lon2-lon1));

  xy[0] = m * cos(lat2) * sin(lon2 - lon1);
  xy[1] = m * (sin(lat2)*cos(lat1) - cos(lat2)*sin(lat1)*cos(lon2-lon1));
}

void Node::updateXYZ(const double xyz[])
{
    this->xyz_coords[0] = xyz[0];
    this->xyz_coords[1] = xyz[1];
    this->xyz_coords[2] = xyz[2];

    cart2sph(xyz_coords, sph_coords);
}

Node * Node::operator+(const Node &other_node)
{
  double new_coords[3];
  Node * new_node;

  new_coords[0] = this->xyz_coords[0] + other_node.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] + other_node.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] + other_node.xyz_coords[2];

  new_node = new Node(new_coords);

  return new_node;
}

bool Node::operator==(const Node &other_node)
{
  if (this->ID == other_node.ID) return true;
  else return false;
}

bool Node::operator!=(const Node &other_node)
{
  if (this->ID == other_node.ID) return false;
  else return true;
}

Node * Node::operator*(const double scalar)
{
    double new_coords[3];
    Node * new_node;

    new_coords[0] = this->xyz_coords[0]*scalar;
    new_coords[1] = this->xyz_coords[1]*scalar;
    new_coords[2] = this->xyz_coords[2]*scalar;

    new_node = new Node(new_coords);

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
  // sph_coords[1] = pi*0.5 - sph_coords[1];

  return *this;
}
