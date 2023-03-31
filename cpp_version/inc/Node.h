//NODE FILE

#ifndef NODE_H_INCDLUDED
#define NODE_H_INCDLUDED

#include <vector>
#include "Face.h"
#include "Vertex.h"
#include "Element.h"

class Node : public Element
{
  private:
    std::string name = "node";
    
  public:
    double area;

    std::vector<Node *> friends_list;        // array pointing to friends
    std::vector<Node *> temp_friends;     // array pointing to temp friends
    std::vector<Face *> face_list;
    std::vector<Vertex *> vertex_list;
    std::vector<double> node_dists;
    std::vector<int> face_dirs;
    std::vector<double> vertex_areas;

    std::vector<Node *> node_ghost_list;
    std::vector<Face *> face_ghost_list;
    std::vector<Vertex *> vertex_ghost_list;

    std::vector<int> updated = std::vector<int> (6, 0);          // have friends been updated?

    std::vector< std::vector<double> > centroids;// = std::vector< std::vector<double> > (6, std::vector<double> (3));

    int friend_num;           // number of surrounding friends
    int dead_num = 0;
    int pentagon = 0;
    int boundary = 0;

    // Constructor takes xyz coords, and usually an ID
    Node(double xyz[], int ID_num=0);

    // Copy constructor
    Node(const Node &n);
    // Node(Element d);/
    Node(const Element &other );
    ~Node();

    Node * operator+(const Node &n);
    Node operator-(const Node &n);
    Node * operator*(const double scalar);
    bool operator==(const Node &n);
    bool operator!=(const Node &n);
    Node & operator=(const Node &n);

    // function adds a node to frieds array
    void addFriend(Node * n);
    void addCentroid(double sph[]);
    void updateCentroidPos(int index, double sph[]);
    void clearCentroids();
    void addTempFriend(Node * n);
    int isFriend(Node * n);
    void addCentroids(std::vector<double> ll_coords);

    bool hasFace(Face * f);

    void updateGhosts();

    void updateArea();

    void orderVertexList();
    void orderFaceList();
    void orderFriendList();
    void updateNodeDists();
    void updateFaceDirs();
    void updateVertexAreas();

    // void printCoords();
};

#endif
