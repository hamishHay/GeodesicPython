//FACE FILE

#ifndef FACE_H_INCDLUDED
#define FACE_H_INCDLUDED

#include <vector>
#include "Element.h"
#include "Node.h"
#include "Vertex.h"
#include <string>


class Node; // Forward delcaration of Node

class Face : public Element
{
  private:
    std::string name = "face";
  public:
    double xyz_normal[3];
    double sph_normal[2];  // [0] is lon direction, [1] is lat direction
    double sph_intersect[3];

    double length;
    double length_intersect;
    double area;

    std::vector<Face *> friends_list1;
    std::vector<Face *> friends_list2;
    Node * n1;
    Node * n2;

    Vertex * v1;
    Vertex * v2;
    

    int friend_num;           // number of surrounding friends
    // int dead_num = 0;
    // int pentagon = 0;
    // int boundary = 0;

    // Constructor takes xyz coords, and usually an ID
    Face(int ID_num, Vertex * vert1, Vertex * vert2, Node * node1, Node * node2);

    bool sharedNode(Node * node1, Node * node2);
    bool hasNode(Node * node1);

    void updateLength(void);
    void updateNormalVec(void);
    void updateCenterPos(void);
    void updateFaceFriends(void);
    void updateIntersectPos(void);
    void updateIntersectLength(void);
    void updateArea(void);

};

#endif
