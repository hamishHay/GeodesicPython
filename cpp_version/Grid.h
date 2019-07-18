//GridFile

#ifndef GRID_H_INCDLUDED
#define GRID_H_INCDLUDED

#include "Grid.h"
#include "Node.h"
#include <vector>

class Grid
{
public:

  Grid(void);

  int recursion_lvl;
  std::vector< std::vector<int> > friends_list;
  std::vector<Node*> node_list;
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
};

#endif
