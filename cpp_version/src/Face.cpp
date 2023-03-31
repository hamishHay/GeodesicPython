#include "Node.h"
#include "Face.h"
#include "Vertex.h"
#include "math_functions.h"
#include <iostream>
#include <algorithm>
#include <vector>

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

        double vec1[2] = {0., 1.};
        double vec2[2];

        double dot_prod;
        double det;

        sph2Map(sph_parent, element->sph_coords, vec2);

        dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1];

        det = vec1[0]*vec2[1] - vec1[1]*vec2[0];

        // this->ang = atan2(dot_prod, det)*180./pi;

        double angle = atan2(det, dot_prod)*180./pi;
        if (angle > 0.0+1e-8) angle -= 360.0;
        this->ang = angle;

    };

    bool operator<( const AngleSort& rhs ) const { return ang < rhs.ang; }
};

Face::Face(int ID_num, Vertex * vert1, Vertex * vert2, Node * node1, Node * node2) : Element(ID_num)
{
    ID = ID_num;

    n1 = node1; // Upwind node
    n2 = node2; // Downwind node

    v1 = vert1;
    v2 = vert2;

    updateCenterPos();
    updateLength();
    updateIntersectLength();
    updateNormalVec();
};

void Face::updateCenterPos(void)
{
    // Find face center coordinates
    midpointBetweenSph(this->v1->sph_coords, this->v2->sph_coords, this->sph_coords);
    
    double sph[3];
    sph[0] = this->sph_coords[0];
    sph[1] = 0.5*pi - this->sph_coords[1];
    sph[2] = this->sph_coords[2];

    
    sph2cart(sph, this->xyz_coords);
};

void Face::updateLength(void)
{
    // Update face length
    this->length = sphericalLength(v1->sph_coords, v2->sph_coords);
}; 

void Face::updateIntersectLength(void)
{
    // Update length between n1 and n2
    this->length_intersect = sphericalLength(n1->sph_coords, n2->sph_coords);
};

void Face::updateArea(void)
{
    this->area = 0.0;
    this->area = sphericalArea(n1->sph_coords, v1->sph_coords, v2->sph_coords);
    this->area += sphericalArea(n2->sph_coords, v1->sph_coords, v2->sph_coords);
};

void Face::updateNormalVec(void)
{
    // Find face normal vector in cartesian components
    normalVectorBetweenXYZ(v1->sph_coords, v2->sph_coords, this->xyz_normal); 

    // Convert to coordinates in spherical lat-lon space
    cart2sphNormalVector(this->xyz_coords, this->xyz_normal, this->sph_normal);

    // Now we need to order the storage order of the nodes 
    // so that n2 is *always* downwind of the normal
    // (This is necessary for the gradient operator)
    double dist1, dist2;
    double face_pos_sph[3];
        
    face_pos_sph[0] = this->sph_coords[0];
    face_pos_sph[1] = this->sph_coords[1];
    face_pos_sph[2] = this->sph_coords[2];

    dist1 = sphericalLength(n2->sph_coords, face_pos_sph);

    face_pos_sph[1] += this->sph_normal[1]*1e-5; // add small change in latitude direction
    face_pos_sph[2] += this->sph_normal[0]*1e-5; // add small change in longitude direction

    dist2 = sphericalLength(n2->sph_coords, face_pos_sph);

    if ( dist2 < dist1 ) {}  // node2 is downwind 
    else {                   // node2 is upwind - swap the order of nodes
        Node * temp = n2;
        n2 = n1;
        n1 = temp;
    }
};

void Face::updateGhosts(void)
{


    if (this->region != n1->region) node_ghost_list.push_back(n1);
    if (this->region != n2->region) node_ghost_list.push_back(n2);

    if (this->region != v1->region) vertex_ghost_list.push_back(v1);
    if (this->region != v2->region) vertex_ghost_list.push_back(v2);
    

    Face * f_friend;
    for (unsigned i=0; i<this->friends_list1.size(); i++)
    {
        f_friend = this->friends_list1[i];

        if (this->region != f_friend->region) face1_ghost_list.push_back(f_friend);
    }

    for (unsigned i=0; i<this->friends_list2.size(); i++)
    {
        f_friend = this->friends_list2[i];
        if (this->region != f_friend->region) face2_ghost_list.push_back(f_friend);
    }

}

// NOTE: This function requires each node's face_list to *already* be sorted. 
void Face::updateFaceFriends(void)
{
    Node * node = n1;

    unsigned start;
    for (unsigned i=0; i<node->face_list.size(); i++)
    {
        Face * face = node->face_list[i];
        if (face->ID == this->ID) {
            start = i;        // Get index of parent face in the node's face list
            break;
        }
    }
    // Now loop over node's faces and add to the list. This way, they are already ordered.
    for (unsigned i=start; i<node->face_list.size()+start; i++)
    {
        Face * face = node->face_list[i%node->face_list.size()];
        if (face->ID != this->ID) this->friends_list1.push_back( face );
    }


    node = n2;
    for (unsigned i=0; i<node->face_list.size(); i++)
    {
        Face * face = node->face_list[i];
        if (face->ID == this->ID) {
            start = i;        // Get index of parent face in the node's face list
            break;
        }
    }
    // Now loop over node's faces and add to the list. This way, they are already ordered.
    for (unsigned i=start; i<node->face_list.size()+start; i++)
    {
        Face * face = node->face_list[i%node->face_list.size()];
        if (face->ID != this->ID) this->friends_list2.push_back( face );
    }

    // Order them properly!
    std::vector<AngleSort<Face>> ordered1;
    // std::cout<<this->ID<<std::endl;
    for (unsigned i=0; i<this->friends_list1.size(); i++) {
        Face * face_friend = this->friends_list1[i];

        // std::cout<<' '<<face_friend->ID;

        ordered1.push_back( AngleSort<Face>()  );
        ordered1[i].element = face_friend;

        double vec1[2] = {0., 1.};
        double vec2[2];

        double dot_prod;
        double det;

        // Get vector from node to parent face (this)
        sph2Map(this->n1->sph_coords, this->sph_coords, vec1);

        // Get vector from node to face friend
        sph2Map(this->n1->sph_coords, face_friend->sph_coords, vec2);

        dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1];

        det = vec1[0]*vec2[1] - vec1[1]*vec2[0];

        double angle = atan2(det, dot_prod)*180./pi;
        if (angle < 0.0) angle += 360.0;

        ordered1[i].ang = angle;
    }

    // std::cout<<std::endl;
    std::sort(ordered1.begin(), ordered1.end());

    for (unsigned i=0; i<this->friends_list1.size(); i++) this->friends_list1[i] = ordered1[i].element;

    std::vector<AngleSort<Face>> ordered2;
    // std::cout<<this->ID<<std::endl;
    for (unsigned i=0; i<this->friends_list2.size(); i++) {
        Face * face_friend = this->friends_list2[i];

        // std::cout<<' '<<face_friend->ID;

        ordered2.push_back( AngleSort<Face>()  );
        ordered2[i].element = face_friend;

        double vec1[2] = {0., 1.};
        double vec2[2];

        double dot_prod;
        double det;

        // Get vector from node to parent face (this)
        sph2Map(this->n2->sph_coords, this->sph_coords, vec1);

        // Get vector from node to face friend
        sph2Map(this->n2->sph_coords, face_friend->sph_coords, vec2);

        dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1];

        det = vec1[0]*vec2[1] - vec1[1]*vec2[0];
        
        double angle = atan2(det, dot_prod)*180./pi;
        if (angle < 0.0) angle += 360.0;

        ordered2[i].ang = angle;
    }

    // std::cout<<std::endl;
    std::sort(ordered2.begin(), ordered2.end());

    for (unsigned i=0; i<this->friends_list2.size(); i++) this->friends_list2[i] = ordered2[i].element;

};

void Face::updateIntersectPos(void)
{
    intersectPointSph(v1->sph_coords, v2->sph_coords, n1->sph_coords, n2->sph_coords, this->sph_intersect);
};

bool Face::sharedNode(Node * node1, Node * node2){
    if ( ( (n1==node1) && (n2==node2) ) || ( (n1==node2) && (n2==node1) )   ) return true;
    else return false;
};

bool Face::hasNode(Node * node){
    if (this->n1 == node) return true;
    else if (this->n2 == node) return true;
    return false;
}
