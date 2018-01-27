//GridFile

#include "Grid.h"
#include "Node.h"
#include <vector>

class Grid
{
public:
  Grid();

  std::vector<Node> nodes;

  void addNode(Node n);
};
