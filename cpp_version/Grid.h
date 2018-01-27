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

  std::vector<Node> nodes;

  void addNode(Node n);
};

#endif
