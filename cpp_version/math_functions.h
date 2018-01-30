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
  sph_coords[2] = phi+pi;

  if ((phi+phi)*180./pi > 359.9) sph_coords[2] = 0.0;
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

  if (fabs(r) < 1e-8)     r = 0.0;
  if (fabs(theta) < 1e-8) theta = 0.0;
  if (fabs(phi) < 1e-8)   phi = 0.0;

  xyz[0] = r*sin(theta)*cos(phi);
  xyz[1] = r*sin(theta)*sin(phi);
  xyz[2] = r*cos(theta);

};

#endif
