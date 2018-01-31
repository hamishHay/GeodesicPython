#include <math.h>
#include <iostream>
# define pi 3.141592653589793238462643383279502884L

#ifndef MATH_FUNCTIONS_H_INCDLUDED
#define MATH_FUNCTIONS_H_INCDLUDED

void inline cart2sph(double xyz[], double sph_coords[]);
void inline cart2sph(double xyz[], double sph_coords[])
{
  double x, y, z;
  double r, theta, phi;

  x = xyz[0];
  y = xyz[1];
  z = xyz[2];

  r = sqrt(x*x + y*y + z*z);
  theta = pi*0.5 - acos(z/r);
  phi = atan2(y,x);


  sph_coords[0] = r;
  sph_coords[1] = theta;
  sph_coords[2] = phi;

  if ((phi)*180./pi > 359.9) sph_coords[2] = 0.0;
};

void inline sph2cart(double sph_coords[], double xyz[], bool rad);
void inline sph2cart(double sph_coords[], double xyz[], bool rad=true)
{
  double r, theta, phi;

  r = sph_coords[0];
  theta = sph_coords[1];
  phi = sph_coords[2];


  if (!rad)
  {
    theta *= pi/180.;
    phi *= pi/180.;
  }

  // if (fabs(r) < 1e-8)     r = 0.0;
  // if (fabs(theta) < 1e-8) theta = 0.0;
  // if (fabs(phi) < 1e-8)   phi = 0.0;

  xyz[0] = r*sin(theta)*cos(phi);
  xyz[1] = r*sin(theta)*sin(phi);
  xyz[2] = r*cos(theta);

  // std::cout<<xyz[0]<<'\t'<<xyz[1]<<'\t'<<xyz[2]<<std::endl;

};

double inline sphericalLength(double sph_c1[], double sph_c2[]);
double inline sphericalLength(double sph_c1[], double sph_c2[])
{
    double length;
    double lat1, lat2, dlon, dlat;

    lat1 = sph_c1[1];
    lat2 = sph_c2[1];
    dlat = lat2 - lat1;
    dlon = sph_c2[2]-sph_c1[2];

    length = sin(dlat*0.5)*sin(dlat*0.5);
    length += cos(lat1)*cos(lat2)*sin(dlon*0.5)*sin(dlon*0.5);
    length = 2.*asin(sqrt(length));

    return length;
}


double inline sphericalArea(double sph_c1[], double sph_c2[], double sph_c3[]);
double inline sphericalArea(double sph_c1[], double sph_c2[], double sph_c3[])
{
    double a, b, c;
    double A, B, C, E;

    a = sphericalLength(sph_c1, sph_c2);
    b = sphericalLength(sph_c2, sph_c3);
    c = sphericalLength(sph_c3, sph_c1);

    A = acos((cos(a) - cos(b)*cos(c))/(sin(b)*sin(c)));
    B = asin(sin(A)*sin(b)/sin(a));
    C = asin(sin(A)*sin(c)/sin(a));

    E = (A + B + C) - pi;

    return fabs(E);
}

// void inline sph2cart(double sph_coords[], double xyz[], bool rad=true)

#endif
