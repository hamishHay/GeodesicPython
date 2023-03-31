#include <math.h>
#include <iostream>
# define pi 3.141592653589793238462643383279502884L

#ifndef MATH_FUNCTIONS_H_INCDLUDED
#define MATH_FUNCTIONS_H_INCDLUDED

void inline sph2Map(double sph1[], double sph2[], double map_xy[]);
void inline sph2Map(double sph1[], double sph2[], double map_xy[])
{
  double m;
  double lat1, lat2, lon1, lon2;

  lat1 = sph1[1];
  lon1 = sph1[2];
  lat2 = sph2[1];
  lon2 = sph2[2];

  m = 2.0 / (1.0 + sin(lat2)*sin(lat1) + cos(lat1)*cos(lat2)*cos(lon2-lon1));

  map_xy[0] = m * cos(lat2) * sin(lon2 - lon1);
  map_xy[1] = m * (sin(lat2)*cos(lat1) - cos(lat2)*sin(lat1)*cos(lon2-lon1));
}

void inline cart2sph(double xyz[], double sph_coords[]);
void inline cart2sph(double xyz[], double sph_coords[])
{
  double x, y, z;
  double r, theta, phi;

  x = xyz[0];
  y = xyz[1];
  z = xyz[2];

  r = sqrt(x*x + y*y + z*z);
  theta = pi*0.5 - acos(z/r); // Converts to latitude
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

  // expects colatitude!
  xyz[0] = r*sin(theta)*cos(phi);
  xyz[1] = r*sin(theta)*sin(phi);
  xyz[2] = r*cos(theta);

  // std::cout<<xyz[0]<<'\t'<<xyz[1]<<'\t'<<xyz[2]<<std::endl;

};

double inline sphericalLength(double sph_c1[], double sph_c2[]);
double inline sphericalLength(double sph_c1[], double sph_c2[])
{
    double length;
    double lat1, lat2, dlon;

    // Expects latitude
    lat1 = sph_c1[1];
    lat2 = sph_c2[1];
    dlon = fabs(sph_c2[2]-sph_c1[2]);

    // length = sin(dlat*0.5)*sin(dlat*0.5);
    // length += cos(lat1)*cos(lat2)*sin(dlon*0.5)*sin(dlon*0.5);
    // length = 2.*asin(sqrt(length));

    length = acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2) * cos(dlon) );

    return length;
};


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
};

void inline crossProduct(double v1[], double v2[], double v1xv2[]);
void inline crossProduct(double v1[], double v2[], double v1xv2[])
{
    v1xv2[0] = v1[1]*v2[2] - v2[1]*v1[2];
    v1xv2[1] = -(v1[0]*v2[2] - v2[0]*v1[2]);
    v1xv2[2] = v1[0]*v2[1] - v2[0]*v1[1];
};

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

    double mag = 0.0;

    for (int k=0; k<3; k++) mag += pow(vcross[k], 2.0);
    mag = sqrt(mag);

    for (int k=0; k<3; k++) vc[k] = -vcross[k]/mag;

};

double inline dotProduct(double v1[], double v2[]);
double inline dotProduct(double v1[], double v2[])
{
    double v1Dotv2;

    v1Dotv2 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

    return v1Dotv2;
};

bool inline isInsideSphericalTriangle(double v1[], double v2[], double v3[], double v[]);
bool inline isInsideSphericalTriangle(double v1[], double v2[], double v3[], double v[])
{
    double n1[3], n2[3], n3[3];
    // double s1[3], s2[3], s3[3];
    // int k;

    crossProduct(v1, v2, n1);
    crossProduct(v2, v3, n2);
    crossProduct(v3, v1, n3);

    // for (k=0; k<3; k++)
    // {
    //     s1[k] = v[k] - v1[k];
    //     s2[k] = v[k] - v2[k];
    //     s3[k] = v[k] - v3[k];
    // }

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
};

bool inline isOnEdge(double v1[], double v2[], double v[]);
bool inline isOnEdge(double v1[], double v2[], double v[])
{
    double n1[3];

    crossProduct(v1, v2, n1);

    if (fabs(dotProduct(v, n1)) < 1e-8) return true;
    // if (dotProduct(s1, n1) > -1e-8 && dotProduct(v, n2) > -1e-8 && dotProduct(v, n3) > -1e-8) return true;
    else return false;
};

void inline getCircularBoundaryPosition(double angularRadius, double offset, double xyz[]);
void inline getCircularBoundaryPosition(double angularRadius, double offset, double xyz[])
{
    double radius = 1.0;
    xyz[0] = radius * cos(angularRadius);
    xyz[1] = radius * sin(angularRadius) * sin(offset);
    xyz[2] = radius * sin(angularRadius) * cos(offset);
};

double inline getAngleBetween(double v1[], double v2[]);
double inline getAngleBetween(double v1[], double v2[])
{
    double mag1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    double mag2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);

    return acos(dotProduct(v1, v2)/(mag1*mag2));
};

// void inline sph2cart(double sph_coords[], double xyz[], bool rad=true)


inline void midpointBetweenSph(double sph1[], double sph2[], double sphc[]);
inline void midpointBetweenSph(double sph1[], double sph2[], double sphc[])
{
  double xyz1[3], xyz2[3];
  double xyzc[3];

  // Expects lat, not colat
  xyz1[0] = cos(sph1[1])*cos(sph1[2]);
  xyz1[1] = cos(sph1[1])*sin(sph1[2]);
  xyz1[2] = sin(sph1[1]);

  xyz2[0] = cos(sph2[1])*cos(sph2[2]);
  xyz2[1] = cos(sph2[1])*sin(sph2[2]);
  xyz2[2] = sin(sph2[1]);

  for (int k=0; k<3; k++) {
      xyzc[k] = 0.5*(xyz1[k]+xyz2[k]);
  }

  sphc[0] = 1.0;
  sphc[1] = atan2(xyzc[2], sqrt(pow(xyzc[0], 2.0) + pow(xyzc[1], 2.0)));
  sphc[2] = atan2(xyzc[1], xyzc[0]);
};

// Function to find the normal unit vector in cartesian space between two points
// with spherical coordinates sph1 and sph2
inline void normalVectorBetweenXYZ(double &, double &, double &);
inline void normalVectorBetweenXYZ(double sph1[], double sph2[], double nxyz[])
{
    double c1[3], c2[3];
    double sph1_temp[3], sph2_temp[3];

    // Convert to colat because why
    // sph1[1] = pi*0.5 - sph1[1];
    // sph2[1] = pi*0.5 - sph2[1];

    sph1_temp[0] = sph1[0];
    sph1_temp[1] = 0.5*pi - sph1[1];
    sph1_temp[2] = sph1[2];

    sph2_temp[0] = sph2[0];
    sph2_temp[1] = 0.5*pi - sph2[1];
    sph2_temp[2] = sph2[2];

    // Get cartesian coords of each spherical coordiate
    sph2cart(sph1_temp, c1);
    sph2cart(sph2_temp, c2);

    // Take cross product to find vector tangent to line between sph1 and sph2
    nxyz[0] = c1[1]*c2[2] - c1[2]*c2[1];
    nxyz[1] = -(c1[0]*c2[2] - c1[2]*c2[0]);
    nxyz[2] = c1[0]*c2[1] - c1[1]*c2[0];

    double mag = sqrt(nxyz[0]*nxyz[0] + nxyz[1]*nxyz[1] + nxyz[2]*nxyz[2]);

    nxyz[0] /= mag;
    nxyz[1] /= mag;
    nxyz[2] /= mag;

    // Convert back, just in case
    // sph1[1] = pi*0.5 - sph1[1];
    // sph2[1] = pi*0.5 - sph2[1];

    // midpointBetweenSph(sph1, sph2, sph_mid);

    // double lat = sph_mid[0];
    // double lon = sph_mid[1];

    // double cxyz[3];

    // // Get cartesian coords of face midpoint
    // sph2cart(cxyz[0], cxyz[1], cxyz[2], 1.0, sph_mid[0], sph_mid[1]);

    // double lonx, lony, lonz;
    // double latx, laty, latz;
    // double zfact = 1.0/sqrt(1 - pow(cxyz[2], 2.0));

    // // longitude unit vector, in cartesian components
    // lonx = zfact * -cxyz[1];
    // lony = zfact * cxyz[0];
    // lonz = 0.0;
    
    // // latitude unit vector, in cartesian components
    // latx = zfact * -cxyz[2]*cxyz[0];
    // laty = zfact * -cxyz[2]*cxyz[1];
    // latz = zfact * (1-cxyz[2]*cxyz[2]);

    // double nlon, nlat;

    // // lat and lon unit vectors 
    // nlon = n[0]*lonx + n[1]*lony + n[2]*lonz;
    // nlat = n[0]*latx + n[1]*laty + n[2]*latz;
};

inline void cart2sphNormalVector(double &, double &, double &);
inline void cart2sphNormalVector(double xyz[], double nxyz[], double nsph[])
{
    // Get cartesian coords of face midpoint
    // sph2cart(xyz, 1.0, sph_mid[0], sph_mid[1]);

    double lonx, lony, lonz;
    double latx, laty, latz;
    double zfact = 1.0/sqrt(1.0 - pow(xyz[2], 2.0));

    // longitude unit vector, in cartesian components
    lonx = zfact * -xyz[1];
    lony = zfact * xyz[0];
    lonz = 0.0;
    
    // latitude unit vector, in cartesian components
    latx = zfact * -xyz[2]*xyz[0];
    laty = zfact * -xyz[2]*xyz[1];
    latz = zfact * (1.0-xyz[2]*xyz[2]);

    double nlon, nlat;

    // lat and lon unit vectors 
    nlon = nxyz[0]*lonx + nxyz[1]*lony + nxyz[2]*lonz;
    nlat = nxyz[0]*latx + nxyz[1]*laty + nxyz[2]*latz;

    nsph[0] = nlon;
    nsph[1] = nlat;

};

inline void intersectPointSph(double &, double &, double &, double &, double &);
inline void intersectPointSph(double sph1[], double sph2[], double sph3[], double sph4[], double sph_int[])
{
    double c1[3], c2[3], c3[3], c4[4], n1[3], n2[3], mp[3];

    // Convert spherical (lat-lon) to cartesian coords 
    c1[0] = cos(sph1[1])*cos(sph1[2]);
    c2[0] = cos(sph2[1])*cos(sph2[2]);
    c3[0] = cos(sph3[1])*cos(sph3[2]);
    c4[0] = cos(sph4[1])*cos(sph4[2]);

    c1[1] = cos(sph1[1])*sin(sph1[2]);
    c2[1] = cos(sph2[1])*sin(sph2[2]);
    c3[1] = cos(sph3[1])*sin(sph3[2]);
    c4[1] = cos(sph4[1])*sin(sph4[2]);

    c1[2] = sin(sph1[1]);
    c2[2] = sin(sph2[1]);
    c3[2] = sin(sph3[1]);
    c4[2] = sin(sph4[1]);

    // Take cross product to find vector tangent to line between sph1 and sph2
    n1[0] = c1[1]*c2[2] - c1[2]*c2[1];
    n1[1] = -(c1[0]*c2[2] - c1[2]*c2[0]);
    n1[2] = c1[0]*c2[1] - c1[1]*c2[0];

    n2[0] = c3[1]*c4[2] - c3[2]*c4[1];
    n2[1] = -(c3[0]*c4[2] - c3[2]*c4[0]);
    n2[2] = c3[0]*c4[1] - c3[1]*c4[0];

    double mag1_r = 1.0/sqrt(n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2]);
    double mag2_r = 1.0/sqrt(n2[0]*n2[0] + n2[1]*n2[1] + n2[2]*n2[2]);

    n1[0] *= mag1_r;
    n1[1] *= mag1_r;
    n1[2] *= mag1_r;

    n2[0] *= mag2_r;
    n2[1] *= mag2_r;
    n2[2] *= mag2_r;

    mp[0] = n1[1]*n2[2] - n2[1]*n1[2];
    mp[1] = n2[0]*n1[2] - n1[0]*n2[2];
    mp[2] = n1[0]*n2[1] - n2[0]*n1[1];

    mag1_r = 1.0/sqrt(mp[0]*mp[0] + mp[1]*mp[1] + mp[2]*mp[2]);
    mp[0] *= mag1_r;
    mp[1] *= mag1_r;
    mp[2] *= mag1_r;

    sph_int[0] = 1.0; 
    sph_int[1] = asin(mp[2]/1.0);
    sph_int[2] = atan2(mp[1], mp[0]);// + pi;

    // if (sph_int[1] < 0.0) sph_int[1] += 2*pi;
}


#endif
