//NODE FILE

#ifndef ELEMENT_H_INCDLUDED
#define ELEMENT_H_INCDLUDED

#include <vector>

class Element
{
  public:
    double xyz_coords[3];     // cartesian coords
    double sph_coords[3];     // spherical coords
    int ID;                   // Global ID
    int RID = 0;              // Local ID
    int region = 0;           // Region that element resides in

    // Constructor takes xyz coords, and usually an ID
    Element(int ID_num);
    Element(double xyz[], int ID_num=0);
    Element(const Element &n);

    ~Element();

    Element * operator+(const Element &n);
    Element operator-(const Element &n);
    Element * operator*(const double scalar);
    bool operator==(const Element &n);
    bool operator!=(const Element &n);
    Element & operator=(const Element &n);

    // project current xyz coords onto sphere of radius r
    void project2Sphere(double r=1.0);

    void updateXYZ(const double xyz[]);

    void transformSph(const double rot);

    // return the coordinates (or copy?) of the Node
    double * getCartCoords();
    double * getSphCoords();
    void getMapCoords(const Element &n, double xy[]);
    double getMagnitude();

    void printCoords();
};

#endif
