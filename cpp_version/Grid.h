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

  void addNode(Node * n);

  void findFriends(void);

  void orderFriends(void);

  void bisectEdges(void);

  void findCentroids(void);

  void shiftNodes(void);

  void saveGrid2File(void);
};

#endif
