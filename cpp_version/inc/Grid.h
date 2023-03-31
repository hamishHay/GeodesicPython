//GridFile

#ifndef GRID_H_INCDLUDED
#define GRID_H_INCDLUDED

#include "Grid.h"
#include "Node.h"
#include "Face.h"
#include "Vertex.h"
#include "H5Cpp.h"
#include <vector>

template <typename T>
struct h5DataArray{

    T * data;
    unsigned size;
    hid_t h5type;
    

    h5DataArray(){};
    h5DataArray(unsigned s, hid_t type_def)
    {
        size = s;
        h5type = type_def;
        data = new T[s];
    };

    ~h5DataArray()
    {
        delete[] data;
    };

};


class Grid
{
    private:
    template<typename T> 
    void saveToHDF5Group(hid_t * group, h5DataArray<T> * data, char const *name);
public:

  Grid(void);

  int recursion_lvl;
  std::vector< std::vector<int> > friends_list;
  std::vector<Node*> node_list;
  std::vector<Face*> face_list;
  std::vector<Vertex*> vertex_list;
  std::vector<std::vector< std::vector<int> > > regions
    = std::vector< std::vector< std::vector< int > > > (5, std::vector<std::vector< int>> (4, std::vector<int> (3) ) );

  std::vector<std::vector< int > > inside_region
    = std::vector< std::vector< int > > (5);

  void addNode(Node * n);

  void defineRegion(int, int, int []);

  void findFriends(void);

  void orderFriends(void);

  void bisectEdges(void);

  void findCentroids(void);

  void shiftNodes(void);

  void orderNodesByRegion(void);

  void applyBoundary(void);

  void refineBoundary(void);

  void twistGrid(void);

  void saveGrid2File(void);

  void createFaces(void);

  void createVertices(void);

  void calculateProperties(void);

  void saveGrid2HDF5(void);
};

#endif
