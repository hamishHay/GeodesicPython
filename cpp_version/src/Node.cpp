#include "Node.h"
#include "Face.h"
#include "math_functions.h"
#include <iostream>
#include <algorithm>
#include <vector>

// Object to contain reference to element and
// the angle between the element and some reference 
// point.
template <typename T>
struct AngleSort{

    T * element;
    double ang;
    
    // Constructor does nothing
    AngleSort() {};

    // Constructor calculates the angle for you, 
    // relative to coordinates of sph_parent
    AngleSort(T * s, double sph_parent[])
    {
        this->element = s;

        double v1[2] = {0., 1.};  // Northward unit vector
        double v2[2];

        double dot_prod;
        double det;

        sph2Map(sph_parent, element->sph_coords, v2);

        dot_prod = v1[0]*v2[0] + v1[1]*v2[1];

        det = v1[0]*v2[1] - v1[1]*v2[0];

        // this->ang = atan2(dot_prod, det)*180./pi;

        double angle = atan2(det, dot_prod)*180./pi;
        if (angle < 0.0) angle += 360.0;
        this->ang = angle;
    };

    bool operator<( const AngleSort& rhs ) const { return ang < rhs.ang; }
};

Node::Node(double xyz[], int ID_num) : Element(xyz, ID_num)
{};


Node::Node(const Node &other_node) : Element(other_node)
{
  this->xyz_coords[0] = other_node.xyz_coords[0];
  this->xyz_coords[1] = other_node.xyz_coords[1];
  this->xyz_coords[2] = other_node.xyz_coords[2];

  this->sph_coords[0] = other_node.sph_coords[0];
  this->sph_coords[1] = other_node.sph_coords[1];
  this->sph_coords[2] = other_node.sph_coords[2];

  this->ID = other_node.ID;
  this->friend_num = other_node.friend_num;

};

void Node::addFriend(Node * n)
{
    friends_list.push_back(n);
    friend_num = friends_list.size();
}

void Node::addCentroid(double  sph[])
{
    std::vector<double> pos(3);
    pos[0] = sph[0];
    pos[1] = sph[1];
    pos[2] = sph[2];

    centroids.push_back( pos );
}

void Node::updateCentroidPos(int index, double sph[])
{
    this->centroids[index][0] = sph[0];
    this->centroids[index][1] = sph[1];
    this->centroids[index][2] = sph[2];
}

void Node::clearCentroids()
{
    centroids.clear();
}

int Node::isFriend(Node * n)
{
    for (unsigned i=0; i<friends_list.size(); i++)
    {
        if (*n == *friends_list[i]) return 1;
    }
    return 0;
}

void Node::addTempFriend(Node * n)
{
    temp_friends.push_back(n);
}


void Node::updateGhosts(void)
{
    Node * n_friend;
    for (unsigned i=0; i<this->friends_list.size(); i++)
    {
        n_friend = this->friends_list[i];

        if (this->region != n_friend->region) node_ghost_list.push_back(n_friend);
    }

    Face * f_friend;
    for (unsigned i=0; i<this->face_list.size(); i++)
    {
        f_friend = this->face_list[i];

        if (this->region != f_friend->region) face_ghost_list.push_back(f_friend);
    }

    Vertex * v_friend;
    for (unsigned i=0; i<this->vertex_list.size(); i++)
    {
        v_friend = this->vertex_list[i];

        if (this->region != v_friend->region) vertex_ghost_list.push_back(v_friend);
    }
}

void Node::orderVertexList(void)
{
    unsigned vert_num = this->vertex_list.size();
    std::vector< AngleSort<Vertex> > ordered;
    ordered.reserve(vert_num);

    for (unsigned j=0; j<vert_num; j++) {
        Vertex * vertex = this->vertex_list[j];
        ordered.push_back( AngleSort<Vertex>(vertex, this->sph_coords) );

    }
    std::sort(ordered.begin(), ordered.end()); // Sorts the vector by angle vetween vertex and this.

    for (unsigned j=0; j<vert_num; j++) this->vertex_list[j] = ordered[j].element;
}

void Node::orderFaceList(void)
{
    unsigned face_num = this->face_list.size();
    std::vector< AngleSort<Face> > ordered;
    ordered.reserve(face_num);

    for (unsigned j=0; j<face_num; j++) {
        Face * face = this->face_list[j];
        ordered.push_back( AngleSort<Face>(face, this->sph_coords) );

    }
    std::sort(ordered.begin(), ordered.end()); // Sorts the vector by angle between face and this.

    for (unsigned j=0; j<face_num; j++) this->face_list[j] = ordered[j].element;
}

void Node::orderFriendList(void)
{
    unsigned f_num = this->friends_list.size();
    std::vector< AngleSort<Node> > ordered;
    ordered.reserve(f_num);

    for (unsigned j=0; j<f_num; j++) {
        Node * node = this->friends_list[j];
        ordered.push_back( AngleSort<Node>(node, this->sph_coords) );

    }
    std::sort(ordered.begin(), ordered.end()); // Sorts the vector by angle between friend node and this.

    for (unsigned j=0; j<f_num; j++) this->friends_list[j] = ordered[j].element;
}

void Node::updateArea(void)
{
    double a = 0.0;

    for (unsigned i=0; i<vertex_list.size(); i++)
    {
        Vertex * v1 = vertex_list[i];
        Vertex * v2 = vertex_list[(i+1)%vertex_list.size()];

        a += sphericalArea(this->sph_coords, v1->sph_coords, v2->sph_coords);
    }

    this->area = a;
}

bool Node::hasFace(Face * other_face)
{
    for (unsigned i=0; i<this->face_list.size(); i++)
    {
        if (other_face == this->face_list[i]) return true;
    }
    return false;
}


void Node::updateVertexAreas(void)
{
    double a;
    this->vertex_areas.reserve(vertex_list.size());
    for (unsigned i=0; i<vertex_list.size(); i++)
    {
        Vertex * vertex = this->vertex_list[i];

        // find the two faces shared by node and vertex
        
        std::vector<Face *> shared_faces;
        for (unsigned k=0; k<3; k++){
            Face * face = vertex->face_list[k];
            if (this->hasFace(face)) shared_faces.push_back(face);
            if (shared_faces.size() == 2) break;
        }

        a = sphericalArea(this->sph_coords, vertex->sph_coords, shared_faces[0]->sph_intersect);
        a += sphericalArea(this->sph_coords, vertex->sph_coords, shared_faces[1]->sph_intersect);

        this->vertex_areas[i] = a;
    }
}

void Node::updateNodeDists(void)
{
    this->node_dists.clear();

    for (unsigned i=0; i<this->friends_list.size(); i++)
    {
        this->node_dists.push_back( sphericalLength(this->sph_coords, this->friends_list[i]->sph_coords) );
    }

};

void Node::updateFaceDirs(void)
{
    this->face_dirs.clear();

    int dir;
    double dist1, dist2;
    double face_pos_sph[3];
    for (unsigned i=0; i<this->face_list.size(); i++)
    {
        Face * face = this->face_list[i];

        face_pos_sph[0] = face->sph_coords[0];
        face_pos_sph[1] = face->sph_coords[1];
        face_pos_sph[2] = face->sph_coords[2];

        dist1 = sphericalLength(this->sph_coords, face_pos_sph);

        face_pos_sph[1] += face->sph_normal[1]*1e-5; // add small change in latitude direction
        face_pos_sph[2] += face->sph_normal[0]*1e-5; // add small change in longitude direction

        dist2 = sphericalLength(this->sph_coords, face_pos_sph);

        // Adding a small vector to face center increased the distance from the node -> n points outwards
        if (dist2 > dist1) dir = 1; 

        // Adding a small vector in the n dir to face center decreased dist to node -> n points inwards
        else dir = -1;

        this->face_dirs.push_back(dir);
    }
};

Node * Node::operator+(const Node &other_node)
{
  double new_coords[3];
  Node * new_node;

  new_coords[0] = this->xyz_coords[0] + other_node.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] + other_node.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] + other_node.xyz_coords[2];

  new_node = new Node(new_coords);

  return new_node;
}

bool Node::operator==(const Node &other_node)
{
  if (this->ID == other_node.ID) return true;
  else return false;
}

bool Node::operator!=(const Node &other_node)
{
  if (this->ID == other_node.ID) return false;
  else return true;
}

Node * Node::operator*(const double scalar)
{
    double new_coords[3];
    Node * new_node;

    new_coords[0] = this->xyz_coords[0]*scalar;
    new_coords[1] = this->xyz_coords[1]*scalar;
    new_coords[2] = this->xyz_coords[2]*scalar;

    new_node = new Node(new_coords);

    return new_node;
}

Node Node::operator-(const Node &other_node)
{
  double new_coords[3];

  new_coords[0] = this->xyz_coords[0] - other_node.xyz_coords[0];
  new_coords[1] = this->xyz_coords[1] - other_node.xyz_coords[1];
  new_coords[2] = this->xyz_coords[2] - other_node.xyz_coords[2];

  Node new_node(new_coords);

  return new_node;
}

Node & Node::operator=(const Node &other_node)
{
  this->xyz_coords[0] = other_node.xyz_coords[0];
  this->xyz_coords[1] = other_node.xyz_coords[1];
  this->xyz_coords[2] = other_node.xyz_coords[2];

  // Maybe we shouldn't do this here?
  cart2sph(xyz_coords, sph_coords);
  // sph_coords[1] = pi*0.5 - sph_coords[1];

  return *this;
}


Node::~Node() {
   // Deallocate the memory that was previously reserved
   //  for this string.
//    delete[] _text;
    // delete this;
}
