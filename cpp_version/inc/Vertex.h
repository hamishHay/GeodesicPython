//FACE FILE

#ifndef VERTEX_H_INCDLUDED
#define VERTEX_H_INCDLUDED

#include <vector>
#include "Element.h"
#include "Node.h"
#include "Face.h"
#include <string>

class Node; // Forward delcaration of Node
class Face;

class Vertex : public Element
{
  private:
    std::string name = "vertex";
  public:
    std::vector<Face *> face_list = std::vector<Face *> (3);        // array pointing to friends
    // std::vector<Vertex *> friends_list = std::vector<Vertex *> (3);
    std::vector<Node *> node_list = std::vector<Node *> (3);
    std::vector<int> face_dirs = std::vector<int> (3);

    int friend_num;           // number of surrounding friends
    double area;
    double subareas[3];
    // int dead_num = 0;
    // int pentagon = 0;
    // int boundary = 0;

    // Constructor takes xyz coords, and usually an ID
    Vertex(int ID_num, double xyz[], Node * node1, Node * node2, Node * node3);

    bool sharedNode(Node * node1, Node * node2, Node * node3);

    void findFaces();
    void updateFaceDirs();
    void updateArea();
    void updateSubAreas();
};

#endif
