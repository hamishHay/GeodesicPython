//NODE FILE

#ifndef REGION_H_INCDLUDED
#define REGION_H_INCDLUDED

#include <vector>
#include <array>

class Region
{
  public:
    std::vector< std::array<double, 3> > xyz_coords;     // cartesian coords
    std::vector< std::array<double, 3> > sph_coords;     // spherical coords
    int ID;                   // Global ID

    std::array<double, 3> sph_center;
    std::array<double, 3> sph_antipode;
    std::array<double, 3> xyz_center;
    std::array<double, 3> xyz_antipode;

    std::vector<std::vector< std::vector<double> > > triangles;

    // loacl ID that element holds in each region the domain
    std::vector<int> region_ID; 

    // Constructor takes xyz coords, and usually an ID
    Region(int ID_num);

    void addVertex(double coords[3]);
    void addTriangle(double sph1[], double sph2[], double sph3[]);
    void updateCenter(void);
    bool isInside(double sph[3]);

    void printCoords(void);
};

#endif
