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

Vertex::Vertex(int ID_num, double xyz[], Node * node1, Node * node2, Node * node3) : Element(xyz, ID_num)
{
  node_list[0] = node1;
  node_list[1] = node2;
  node_list[2] = node3;

  // Sort the nodes into ascending order 
  std::sort( node_list.begin( ), node_list.end( ), [ ]( const auto& lhs, const auto& rhs )
  {
   return lhs->ID < rhs->ID;
  });

  cart2sph(xyz_coords, sph_coords);


  // Sort nodes in clockwise order! Important for calculating 
  // Voronoi centres!
  std::vector<AngleSort<Node>> ordered;
    // std::cout<<this->ID<<std::endl;
    for (unsigned i=0; i<this->node_list.size(); i++) {
        Node * node = this->node_list[i];

        // std::cout<<' '<<face_friend->ID;

        ordered.push_back( AngleSort<Node>(node, this->sph_coords));
    }

    std::sort(ordered.begin(), ordered.end());

    for (unsigned k=0; k<3; k++) node_list[k] = ordered[k].element; 

};

bool Vertex::sharedNode(Node * node1, Node * node2, Node * node3)
{
    std::vector<Node *> new_node_list(3);
    
    new_node_list[0] = node1;
    new_node_list[1] = node2;
    new_node_list[2] = node3;

    // Sort the nodes into ascending order 
    std::vector<AngleSort<Node>> ordered;
    // std::cout<<this->ID<<std::endl;
    for (unsigned i=0; i<new_node_list.size(); i++) {
        Node * node = new_node_list[i];

        // std::cout<<' '<<face_friend->ID;

        ordered.push_back( AngleSort<Node>(node, this->sph_coords));
    }

    std::sort(ordered.begin(), ordered.end());

    for (unsigned k=0; k<3; k++) new_node_list[k] = ordered[k].element; 
    
    if (    (new_node_list[0] == node_list[0]) 
         && (new_node_list[1] == node_list[1])
         && (new_node_list[2] == node_list[2]) )
    {
        return true;
    } 
    else return false;
   
};

// Loop through the three nodes to find the common faces shared by 
// the vertex
void Vertex::findFaces(void)
{
    for (unsigned k=0; k<3; k++)
    {
        bool added=false;
        Node * n1 = this->node_list[k      ];
        Node * n2 = this->node_list[(k+1)%3];
        for (unsigned i=0; i<n1->face_list.size(); i++)
        {
            for (unsigned j=0; j<n2->face_list.size(); j++)
            {
                if (n1->face_list[i]->ID == n2->face_list[j]->ID) 
                {
                    face_list[k] = n1->face_list[i] ;
                    added = true;
                    break;
                }
            }
            if (added) break;
        }
    }

    // sort the face list in ascending order
    std::sort( face_list.begin( ), face_list.end( ), [ ]( const auto& lhs, const auto& rhs )
    {
    return lhs->ID < rhs->ID;
    });
    
};

// Loop through the three nodes to find the common faces shared by 
// the vertex
void Vertex::updateFaceDirs(void)
{
    for (unsigned k=0; k<3; k++)
    {
        Face * face = face_list[k];

        double vec[2];
        double normal[2];
        double cross;

        // Get vector from face center to verex pos.
        sph2Map(face->sph_coords, this->sph_coords, vec);

        // // Change vector to point from vertex to face center
        // vec[0] *= -1;
        // vec[1] *= -1;

        normal[0] = face->sph_normal[0];
        normal[1] = face->sph_normal[1];

        cross = vec[0]*normal[1] - normal[0]*vec[1];
        
        if (cross > 0.0) this->face_dirs[k] = 1; // face points anti-clockwise vertex
        else this->face_dirs[k] = -1;            // face points clockwise around vertex
    }
};

void Vertex::updateArea(void)
{
    this->area = sphericalArea(node_list[0]->sph_coords, node_list[1]->sph_coords, node_list[2]->sph_coords);
    // for (unsigned k=0; k<3; k++) {

    // }
}

void Vertex::updateInterpolationWeights(void)
{
    
    for (unsigned k=0; k<3; k++) {
        interp_weights[k] = sphericalArea(this->sph_coords, node_list[(k+1)%3]->sph_coords, node_list[(k+2)%3]->sph_coords)/this->area;
    }
}


void Vertex::updateSubAreas()
{
    for (unsigned k=0; k<3; k++)
    {
        Node * node = this->node_list[k];

        std::vector<Face *> shared_faces;
        for (unsigned i=0; i<face_list.size(); i++)
        {
            Face * face = face_list[i];
            if (node->hasFace(face)) shared_faces.push_back(face);

            if (shared_faces.size() == 2) break;
        }

        this->subareas[k] =   sphericalArea(node->sph_coords, this->sph_coords, shared_faces[0]->sph_intersect)
                            + sphericalArea(node->sph_coords, this->sph_coords, shared_faces[1]->sph_intersect); 
    }
}

void Vertex::updateGhosts()
{
    Node * n_friend;
    for (unsigned i=0; i<this->node_list.size(); i++)
    {
        n_friend = this->node_list[i];

        if (this->region != n_friend->region) this->node_ghost_list.push_back(n_friend);
    }

    Face * f_friend;
    for (unsigned i=0; i<this->face_list.size(); i++)
    {
        f_friend = this->face_list[i];

        if (this->region != f_friend->region) this->face_ghost_list.push_back(f_friend);
    }

};

void Vertex::updatePosition(bool sph)
{
    voronoiCenter(node_list[0]->xyz_coords, node_list[1]->xyz_coords, node_list[2]->xyz_coords, this->xyz_coords);
    if (sph) cart2sph(this->xyz_coords, this->sph_coords);
}


