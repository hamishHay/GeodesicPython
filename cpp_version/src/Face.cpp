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

        // this->ang = atan2(dot_prod, det)*180./_PI;

        double angle = atan2(det, dot_prod)*180./_PI;
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
    sph[1] = this->sph_coords[1];
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

        double angle = atan2(det, dot_prod)*180./_PI;
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
        
        double angle = atan2(det, dot_prod)*180./_PI;
        if (angle < 0.0) angle += 360.0;

        ordered2[i].ang = angle;
    }

    // std::cout<<std::endl;
    std::sort(ordered2.begin(), ordered2.end());

    for (unsigned i=0; i<this->friends_list2.size(); i++) this->friends_list2[i] = ordered2[i].element;

};

// This function relies on n1 being the upwind node
// and n2 being the downwind node
void Face::updateInterpolationWeights(void)
{
    // int ne, tev;
    double ne, tev;
    Face * face_e1, * face_e2;
    Vertex * shared_v;

    face_e1 = this;

    weights1.clear();
    weights2.clear();
    // weights1 = std::vector<double>(friends_list1.size());
    // weights2 = std::vector<double>(friends_list2.size());
    

    // list 1 for x=0, list 2 for x=1
    Node * node;
    for (unsigned x=0; x<2; x++) {
        tev = 0;
        if (x==0) node = n1;
        else      node = n2;

        if (x==0) face_e2 = friends_list1[0];
        else      face_e2 = friends_list2[0];

            // Find shared vertex with first face friend

        // std::cout<<face_e1->v1-ID<<' '<<face_e1->v2-ID<<' '<<face_e2->v1-ID<<' '<<face_e1->v1-ID<<std::endl;
        if (face_e1->v1 == face_e2->v1)         shared_v = face_e1->v1;
        else if (face_e1->v1 == face_e2->v2)    shared_v = face_e1->v1;
        else                                    shared_v = face_e1->v2;

        // Find the tev indicator associated with the shared 
        // vertex and the first friend
        for (unsigned k=0; k<3; k++) {
            if (shared_v->face_list[k] == face_e1) {
                tev = (double)shared_v->face_dirs[k];
                // std::cout<<this->ID<<' '<<shared_v->ID<<' '<<k<<' '<<tev<<std::endl;
                break;
            }
        }

        

        unsigned fnum;
        if (x==0) fnum = friends_list1.size();
        else      fnum = friends_list2.size();
        for (unsigned i=0; i<fnum; i++) {
            ne = 0.0;
            // Calculate wee'
            double wee = 0.0;
            if (x==0) face_e2 = friends_list1[i];
            else      face_e2 = friends_list2[i];

            // Find direction of the first face in the sum, e'
            if (face_e2->n1 == node) ne = 1;     // face points outwards
            else ne = -1;                            // face point inwards

            for (unsigned j=0; j<node->face_list.size(); j++) {
                if (node->face_list[j] == face_e2) {
                    // std::cout<<ne<<' '<<node->face_dirs[j]<<std::endl;
                    ne = (double)node->face_dirs[j];
                }
            }

            for (int j=(int)i; j>=0; j--) {
                Face * face1, *face2;
                if (x==0) face2 = friends_list1[j];
                else      face2 = friends_list2[j];

                if (j==0) {
                    face1 = this;
                }
                else {
                    if (x==0) face1 = friends_list1[j-1];
                    else      face1 = friends_list2[j-1];
                }
                
                if (face1->v1 == face2->v1)         shared_v = face1->v1;
                else if (face1->v1 == face2->v2)    shared_v = face1->v1;
                else                                shared_v = face1->v2;

                for (unsigned k=0; k<3; k++){
                    if (shared_v->node_list[k] == node) {

                        wee += shared_v->subareas[k];
                        // std::cout<<"    "<<j<<' '<<wee/n1->area - 0.5<<' '<<shared_v->subareas[k]<<std::endl;
                        break; 
                    }
                }

                // std::cout<<shared_v->ID<<std::endl;
                
            }

            

            // std::cout<<i<<' '<<x<<' '<<n1->ID<<' '<<n2->ID<<' '<<tev*ne*wee/n1->area-0.5<<' '<<tev<<' '<<ne<<std::endl;

            double weight = 0.0;
            double cv_area = 0.0;
            double tick = -tev*ne;
            if (x==0) cv_area = 1.0/n1->area;
            else cv_area = 1.0/n2->area;
            
            // std::cout<<tev<<'*'<<ne<<'*'<<wee<<'*'<<cv_area<<" - 0.5    "<<weight<<std::endl;

            weight = -tev*ne*(wee*cv_area - 0.5);// - 0.5;
            // weight = weight - 0.5;
            // std::cout<<-tev<<'*'<<ne<<'*'<<wee<<'*'<<cv_area<<" - 0.5    "<<weight<<std::endl;

            // std::cout<<this->ID<<' '<<i<<' '<<face_e2->ID<<' '<<weight<<std::endl;

            


            
            // weight *= -tev * ne;
            // weight -= 0.5;
            // wee = -tev * ne * wee/n1->area;
            // std::cout<<i<<' '<<weight;
            if (x==0) weights1.push_back( weight );
            else      weights2.push_back( weight );

            // if (x==0) {
            //     std::cout<<' '<<i<<' '<<weights1[i];
            // }
            // else std::cout<<' '<<i<<' '<<weights2[i];

            
        

        }
        // std::cout<<std::endl<<std::endl;
        

    }

    


    
}

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
