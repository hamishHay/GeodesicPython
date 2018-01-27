#include <math.h>
# define pi 3.141592653589793238462643383279502884L

void cart2sph(double xyz[], double sph_coords[])
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

  if (phi*180./pi > 359.9) sph_coords[2] = 0.0;
};
