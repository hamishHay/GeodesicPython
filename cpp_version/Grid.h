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

  std::vector< std::vector<int> > friends_list;
  std::vector<Node*> node_list;

  void addNode(Node * n);

  void findFriends(void);

  void bisectEdges(void);
};

#endif
