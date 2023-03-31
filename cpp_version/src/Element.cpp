#include "Element.h"
#include "Node.h"
#include "Face.h"
#include "math_functions.h"
#include <iostream>
#include <algorithm>
#include <vector>

// struct AngleSortVert{
//   double ang;
//   Vertex * vertex;

//   bool operator<( const AngleSortVert& rhs ) const { return ang < rhs.ang; }
// };

// struct AngleSortFace{
//   double ang;
//   Face * face;

//   bool operator<( const AngleSortFace& rhs ) const { return ang < rhs.ang; }
// };

Element::Element(int ID_num)//, std::string &etype)
{
  ID = ID_num;
//   element_type = etype;
};

Element::Element(double xyz[], int ID_num)//, std::string &etype)
{
  xyz_coords[0] = xyz[0];
  xyz_coords[1] = xyz[1];
  xyz_coords[2] = xyz[2];
  ID = ID_num;

  cart2sph(xyz_coords, sph_coords);

//   element_type = etype;
};

// Element::Element(double xyz[])
// {
//   xyz_coords[0] = xyz[0];
//   xyz_coords[1] = xyz[1];
//   xyz_coords[2] = xyz[2];

//   cart2sph(xyz_coords, sph_coords);
// };

Element::Element(const Element &other_element)
{
  this->xyz_coords[0] = other_element.xyz_coords[0];
  this->xyz_coords[1] = other_element.xyz_coords[1];
  this->xyz_coords[2] = other_element.xyz_coords[2];

  this->sph_coords[0] = other_element.sph_coords[0];
  this->sph_coords[1] = other_element.sph_coords[1];
  this->sph_coords[2] = other_element.sph_coords[2];

  this->ID = other_element.ID;
};

void Element::transformSph(const double rot)
{
    double x = this->xyz_coords[0];
    double y = this->xyz_coords[1];

    this->xyz_coords[0] = cos(rot)*x + sin(rot)*y;
    this->xyz_coords[1] = -sin(rot)*x + cos(rot)*y;

    cart2sph(xyz_coords, sph_coords);
}

double Element::getMagnitude()
{
    double mag;

    mag = sqrt(this->xyz_coords[0]*this->xyz_coords[0]
              +this->xyz_coords[1]*this->xyz_coords[1]
              +this->xyz_coords[2]*this->xyz_coords[2]);

    return mag;
}

void Element::project2Sphere(double r)
{
    double mag;

    mag = this->getMagnitude();

    this->xyz_coords[0] /= mag;
    this->xyz_coords[1] /= mag;
    this->xyz_coords[2] /= mag;

    // Maybe we shouldn't do this here?
    cart2sph(xyz_coords, sph_coords);
}

void Element::printCoords()
{
    std::cout<<"Coordinates of node "<<this->ID<<": \n";
    std::cout<<" lat: "<<this->sph_coords[1]*180./pi<<", ";
    std::cout<<" lon: "<<this->sph_coords[2]*180./pi<<std::endl;
}

void Element::getMapCoords(const Element &center_node, double xy[])
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

void Element::updateXYZ(const double xyz[])
{
    this->xyz_coords[0] = xyz[0];
    this->xyz_coords[1] = xyz[1];
    this->xyz_coords[2] = xyz[2];

    cart2sph(xyz_coords, sph_coords);
}

Element * Element::operator+(const Element &other_element)
{
  double new_coords[3];
  Element * new_element;

  new_coords[0] = this->xyz_coords[0] + other_element.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] + other_element.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] + other_element.xyz_coords[2];

  new_element = new Element(new_coords);

  return new_element;
}

bool Element::operator==(const Element &other_element)
{
  if (this->ID == other_element.ID) return true;
  else return false;
}

bool Element::operator!=(const Element &other_element)
{
  if (this->ID == other_element.ID) return false;
  else return true;
}

Element * Element::operator*(const double scalar)
{
    double new_coords[3];
    Element * new_element;

    new_coords[0] = this->xyz_coords[0]*scalar;
    new_coords[1] = this->xyz_coords[1]*scalar;
    new_coords[2] = this->xyz_coords[2]*scalar;

    new_element = new Element(new_coords);

    return new_element;
}

Element Element::operator-(const Element &other_element)
{
  double new_coords[3];

  new_coords[0] = this->xyz_coords[0] - other_element.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] - other_element.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] - other_element.xyz_coords[2];

  Element new_element(new_coords);

  return new_element;
}

Element & Element::operator=(const Element &other_element)
{
  this->xyz_coords[0] = other_element.xyz_coords[0];
  this->xyz_coords[1] = other_element.xyz_coords[1];
  this->xyz_coords[2] = other_element.xyz_coords[2];

  // Maybe we shouldn't do this here?
  cart2sph(xyz_coords, sph_coords);
  // sph_coords[1] = pi*0.5 - sph_coords[1];

  return *this;
}

Element::~Element() {
   // Deallocate the memory that was previously reserved
   //  for this string.
//    delete[] _text;
    // delete this;
}
