//NODE FILE

class Node
{
  public:
    double xyz_coords[3];     // cartesian coords
    double sph_coords[3];     // sphericala coords
    double ll_coords[3];      // lat-lon coords

    Node * friends[6];        // array pointing to friends
    Node * temp_friends[6];   // array pointing to temp friends

    bool updated[6];          // have friends been updated?

    int friend_num;           // number of surrounding friends 

    // Constructor takes xyz coords, and usually an ID
    Node(double x, double y, double z, int ID=0);

    Node operator+(const Node &n);
    Node operator-(const Node &n);
    Node operator=(const Node &n);

    // function adds a node to frieds array
    int addFriend(Node &n);
    int addTempFriend(Node &n);

    // project current xyz coords onto sphere of radius r
    void project2Sphere(double radius);

    // return the coordinates (or copy?) of the Node
    double * getCartCoords();
    double * getSphCoords();
}
