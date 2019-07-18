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
    dlon = fabs(sph_c2[2]-sph_c1[2]);

    // length = sin(dlat*0.5)*sin(dlat*0.5);
    // length += cos(lat1)*cos(lat2)*sin(dlon*0.5)*sin(dlon*0.5);
    // length = 2.*asin(sqrt(length));

    length = acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2) * cos(dlon) );

    return length;
}


double inline sphericalArea(double sph_c1[], double sph_c2[], double sph_c3[]);
double inline sphericalArea(double sph_c1[], double sph_c2[], double sph_c3[])
{
    double a, b, c;
    double A, B, C, E;

    a = fabs(sphericalLength(sph_c1, sph_c2));
    b = fabs(sphericalLength(sph_c2, sph_c3));
    c = fabs(sphericalLength(sph_c3, sph_c1));

    // A = fabs( acos((cos(a) - cos(b)*cos(c))/(sin(b)*sin(c))));
    // B = fabs( asin(sin(A)*sin(b)/sin(a)));
    // C = fabs( asin(sin(A)*sin(c)/sin(a)));

    C = fabs( acos((cos(c) - cos(b)*cos(a))/(sin(b)*sin(a))));
    A = fabs( acos((cos(a) - cos(b)*cos(c))/(sin(b)*sin(c))));
    B = fabs( acos((cos(b) - cos(a)*cos(c))/(sin(a)*sin(c))));
    // B = fabs( asin(sin(A)*sin(b)/sin(a)));
    // C = fabs( asin(sin(A)*sin(c)/sin(a)));

    E = (A + B + C) - pi;

    return fabs(E);
}

void inline crossProduct(double v1[], double v2[], double v1xv2[]);
void inline crossProduct(double v1[], double v2[], double v1xv2[])
{
    v1xv2[0] = v1[1]*v2[2] - v2[1]*v1[2];
    v1xv2[1] = -(v1[0]*v2[2] - v2[0]*v1[2]);
    v1xv2[2] = v1[0]*v2[1] - v2[0]*v1[1];
}

void inline voronoiCenter(double v1[], double v2[], double v3[], double vc[]);
void inline voronoiCenter(double v1[], double v2[], double v3[], double vc[])
{
    double v11[3], v12[3];
    double vcross[3];

    for (int k=0; k<3; k++)
    {
        v11[k] = v2[k]-v1[k];
        v12[k] = v3[k]-v1[k];
    }
    crossProduct(v11, v12, vcross);

    double mag;

    for (int k=0; k<3; k++) mag += pow(vcross[k], 2.0);
    mag = sqrt(mag);

    for (int k=0; k<3; k++) vc[k] = -vcross[k]/mag;

}

double inline dotProduct(double v1[], double v2[]);
double inline dotProduct(double v1[], double v2[])
{
    double v1Dotv2;

    v1Dotv2 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

    return v1Dotv2;
}

bool inline isInsideSphericalTriangle(double v1[], double v2[], double v3[], double v[]);
bool inline isInsideSphericalTriangle(double v1[], double v2[], double v3[], double v[])
{
    double n1[3], n2[3], n3[3];
    double s1[3], s2[3], s3[3];
    int k;

    crossProduct(v1, v2, n1);
    crossProduct(v2, v3, n2);
    crossProduct(v3, v1, n3);

    for (k=0; k<3; k++)
    {
        s1[k] = v[k] - v1[k];
        s2[k] = v[k] - v2[k];
        s3[k] = v[k] - v3[k];
    }

    // if (v[2] < 0)
    // {
    //     for (k=0; k<3; k++)
    //     {
    //         n1[k] = -n1[k];
    //         n2[k] = -n2[k];
    //         n3[k] = -n3[k];
    //     }
    // }

    // std::cout<<dotProduct(v, n1)<<'\t'<<dotProduct(v, n2)<<'\t'<<dotProduct(v, n3)<<std::endl;
    if (dotProduct(v, n1) < 1e-8 && dotProduct(v, n2) < 1e-8 && dotProduct(v, n3) < 1e-8) return true;
    // if (dotProduct(s1, n1) > -1e-8 && dotProduct(v, n2) > -1e-8 && dotProduct(v, n3) > -1e-8) return true;
    else return false;
}

bool inline isOnEdge(double v1[], double v2[], double v[]);
bool inline isOnEdge(double v1[], double v2[], double v[])
{
    double n1[3];
    int k;

    crossProduct(v1, v2, n1);

    if (fabs(dotProduct(v, n1)) < 1e-8) return true;
    // if (dotProduct(s1, n1) > -1e-8 && dotProduct(v, n2) > -1e-8 && dotProduct(v, n3) > -1e-8) return true;
    else return false;
}

void inline getCircularBoundaryPosition(double angularRadius, double offset, double xyz[]);
void inline getCircularBoundaryPosition(double angularRadius, double offset, double xyz[])
{
    double radius = 1.0;
    xyz[0] = radius * cos(angularRadius);
    xyz[1] = radius * sin(angularRadius) * sin(offset);
    xyz[2] = radius * sin(angularRadius) * cos(offset);
}

double inline getAngleBetween(double v1[], double v2[]);
double inline getAngleBetween(double v1[], double v2[])
{
    double mag1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    double mag2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);

    return acos(dotProduct(v1, v2)/(mag1*mag2));
}

// void inline sph2cart(double sph_coords[], double xyz[], bool rad=true)

#endif
