//NODE FILE

#ifndef NODE_H_INCDLUDED
#define NODE_H_INCDLUDED

#include <vector>

class Node
{
  public:
    double xyz_coords[3];     // cartesian coords
    double sph_coords[3];     // sphericala coords
    double ll_coords[3];      // lat-lon coords
    int ID;

    std::vector<Node *> friends_list;        // array pointing to friends
    std::vector<Node *> temp_friends;     // array pointing to temp friends

    std::vector<int> updated = std::vector<int> (6, 0);          // have friends been updated?

    std::vector< std::vector<double> > centroids = std::vector< std::vector<double> > (6, std::vector<double> (3));

    int friend_num;           // number of surrounding friends
    int dead_num = 0;
    int pentagon = 0;
    int boundary = 0;

    // Constructor takes xyz coords, and usually an ID
    Node(double xyz[], int ID_num=0);

    // Copy constructor
    Node(const Node &n);

    Node * operator+(const Node &n);
    Node operator-(const Node &n);
    Node * operator*(const double scalar);
    bool operator==(const Node &n);
    bool operator!=(const Node &n);
    Node & operator=(const Node &n);

    // function adds a node to frieds array
    void addFriend(Node * n);
    void addTempFriend(Node * n);
    int isFriend(Node * n);
    void addCentroids(std::vector<double> ll_coords);

    void orderFriends();

    // project current xyz coords onto sphere of radius r
    void project2Sphere(double r=1.0);

    void updateXYZ(const double xyz[]);

    void transformSph(const double rot);

    // return the coordinates (or copy?) of the Node
    double * getCartCoords();
    double * getSphCoords();
    void getMapCoords(const Node &n, double xy[]);
    double getMagnitude();

    void printCoords();
};

#endif
