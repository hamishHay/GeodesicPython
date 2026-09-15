#include "Node.h"
#include "Face.h"
#include "Vertex.h"
#include "Region.h"
#include "math_functions.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>

// struct AngleSort{

//     std::vector<std::array<double, 3>> * element;
//     double ang;
    
//     // Constructor does nothing
//     AngleSort() {};

//     // Constructor calculates the angle for you, 
//     // relative to coordinates of sph_parent
//     AngleSort(T * s, double sph_parent[])
//     {
//         this->element = s;

//         double vec1[2] = {0., 1.};
//         double vec2[2];

//         double dot_prod;
//         double det;

//         sph2Map(sph_parent, element->sph_coords, vec2);

//         dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1];

//         det = vec1[0]*vec2[1] - vec1[1]*vec2[0];

//         // this->ang = atan2(dot_prod, det)*180./_PI;

//         double angle = atan2(det, dot_prod)*180./_PI;
//         if (angle > 0.0+1e-8) angle -= 360.0;
//         this->ang = angle;

//     };

//     bool operator<( const AngleSort& rhs ) const { return ang < rhs.ang; }
// };

Region::Region(int ID_num)
{
    ID = ID_num;
};

void Region::addVertex(double coords[3]) {
    std::array<double, 3> pos = {coords[0], coords[1], coords[2]}; 

    sph_coords.push_back(pos);

    
    double r, lat, lon;
    r = pos[0];
    lat = pos[1];
    lon = pos[2];

    double xyz[3];
    // sph2cart(sph, xyz);


    xyz[0] = r * cos(lat) * cos(lon);
    xyz[1] = r * cos(lat) * sin(lon);
    xyz[2] = r * sin(lat);


    std::array<double, 3> pos2 = {xyz[0], xyz[1], xyz[2]};

    xyz_coords.push_back(pos2);
}

void Region::addTriangle(double sph1[], double sph2[], double sph3[]) {
    // std::array<double, 3> pos = {coords[0], coords[1], coords[2]}; 

    // sph_coords.push_back(pos);

    
    double r1, lat1, lon1;
    double r2, lat2, lon2;
    double r3, lat3, lon3;

    r1 = sph1[0]; lat1 = sph1[1]; lon1 = sph1[2];
    r2 = sph2[0]; lat2 = sph2[1]; lon2 = sph2[2];
    r3 = sph3[0]; lat3 = sph3[1]; lon3 = sph3[2];

    // std::cout<<lat3<<' '<<sin(lat3)<<" LAT HERE"<<std::endl;
    

    double xyz1[3], xyz2[3], xyz3[3];
    // sph2cart(sph, xyz);


    xyz1[0] = r1 * cos(lat1) * cos(lon1);
    xyz1[1] = r1 * cos(lat1) * sin(lon1);
    xyz1[2] = r1 * sin(lat1);
    xyz2[0] = r2 * cos(lat2) * cos(lon2);
    xyz2[1] = r2 * cos(lat2) * sin(lon2);
    xyz2[2] = r2 * sin(lat2);
    xyz3[0] = r3 * cos(lat3) * cos(lon3);
    xyz3[1] = r3 * cos(lat3) * sin(lon3);
    xyz3[2] = r3 * sin(lat3);


    std::array<double, 3> pos1 = {xyz1[0], xyz1[1], xyz1[2]};
    std::array<double, 3> pos2 = {xyz2[0], xyz2[1], xyz2[2]};
    std::array<double, 3> pos3 = {xyz3[0], xyz3[1], xyz3[2]};

    // xyz_coords.push_back(pos2);

    std::vector<std::vector<double>> triangle = std::vector<std::vector<double>>(3);

    triangle[0].push_back(xyz1[0]);
    triangle[0].push_back(xyz1[1]);
    triangle[0].push_back(xyz1[2]);
    triangle[1].push_back(xyz2[0]);
    triangle[1].push_back(xyz2[1]);
    triangle[1].push_back(xyz2[2]);
    triangle[2].push_back(xyz3[0]);
    triangle[2].push_back(xyz3[1]);
    triangle[2].push_back(xyz3[2]);

    // std::cout<<triangle[2][1]<<" LAT HERE"<<std::endl;
    triangles.push_back(triangle);


}

void Region::updateCenter(void) {
    double sumx=0., sumy=0., sumz=0.;
    for (unsigned k=0; k<xyz_coords.size(); k++) {
        sumx += xyz_coords[k][0];
        sumy += xyz_coords[k][1];
        sumz += xyz_coords[k][2];
    }

    xyz_center[0] = sumx/xyz_coords.size();
    xyz_center[1] = sumy/xyz_coords.size();
    xyz_center[2] = sumz/xyz_coords.size();

    cart2sph(&xyz_center[0], &sph_center[0]);
};

bool Region::isInside(double p4_sph[3]) {
    // xyz_center[0]
    // double p1[3], p2[3], p3[3];
    
    double v1[3], v2[3], T[3];
    // double d1[3], d2[3], d4[3], d3[3];
    // double sph1[3], sph2[3];
    // // double p1_sph[3], p2_sph[3], p3_sph[3], p4_sph[3];
    // double l11, l12, l1T;
    // double l21, l22, l2T;
    // double sum1, sum2;

    double r, lat, lon;
    r = p4_sph[0];
    lat = p4_sph[1];
    lon = p4_sph[2];

    double pt[3];
    pt[0] = r * cos(lat) * cos(lon);
    pt[1] = r * cos(lat) * sin(lon);
    pt[2] = r * sin(lat);


    // // p3 = &xyz_antipode[0];
    // p3[0] = xyz_center[0];
    // p3[1] = xyz_center[1];
    // p3[2] = xyz_center[2];


    int num_intersects = 0;

    for (unsigned k=0; k<triangles.size(); k++) {
        double * p1, * p2, * p3;

        // Get coords of great circle 
        p1 = &triangles[k][0][0];
        p2 = &triangles[k][1][0];
        p3 = &triangles[k][2][0];

        // for (unsigned i=0; i<3; i++) {
        //     std::cout<<"   x "<<triangles[k][i][0];
        //     std::cout<<"   y "<<triangles[k][i][1];
        //     std::cout<<"   z "<<triangles[k][i][2]<<std::endl;
        // }

        if (isInsideSphericalTriangle(p1, p2, p3, pt)) return true;
    }

    // for (unsigned k=0; k<xyz_coords.size(); k++) {
    //     double * p1, * p2;

    //     // Get coords of great circle 
    //     p1 = &xyz_coords[k][0];
    //     p2 = &xyz_coords[(k+1)%xyz_coords.size()][0];

    //     p1[0] = xyz_coords[k][0];
    //     p1[1] = xyz_coords[k][1];
    //     p1[2] = xyz_coords[k][2];

    //     p2[0] = xyz_coords[(k+1)%xyz_coords.size()][0];
    //     p2[1] = xyz_coords[(k+1)%xyz_coords.size()][1];
    //     p2[2] = xyz_coords[(k+1)%xyz_coords.size()][2];

    //     if (isInsideSphericalTriangle(p1, p2, p3, pt)) return true;

    // //     crossProduct(p1, p2, v1);
    // //     crossProduct(p3, p4, v2);

        
    // //     double mag1 = 1.0/sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    // //     double mag2 = 1.0/sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
    // //     v1[0] = v1[0]*mag1; v1[1] = v1[1]*mag1; v1[2] = v1[2]*mag1;
    // //     v2[0] = v2[0]*mag2; v2[1] = v2[1]*mag2; v2[2] = v2[2]*mag2;

    // //     crossProduct(v1, v2, d1);
    // //     mag1 = 1.0/sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]);
    // //     d1[0] = d1[0]*mag1; d1[1] = d1[1]*mag1; d1[2] = d1[2]*mag1;
    // //     d2[0] = -d1[0]; d2[1] = -d1[1]; d2[2] = -d2[2];


    // //     cart2sph(d1, sph1);
    // //     cart2sph(d2, sph2);
    // //     cart2sph(p1, p1_sph);
    // //     cart2sph(p2, p2_sph);
    // //     cart2sph(p3, p3_sph);
    // //     cart2sph(p4, p4_sph);

    // //     l1T = sphericalLength(p1_sph, p2_sph);
    // //     l11 = sphericalLength(p1_sph, sph1);
    // //     l12 = sphericalLength(sph1, p2_sph);

    // //     l2T = sphericalLength(p3_sph, p4_sph);
    // //     l21 = sphericalLength(p3_sph, sph1);
    // //     l22 = sphericalLength(sph1, p4_sph);

    // //     sum1 = fabs(l1T - l11 - l12);
    // //     sum2 = fabs(l2T - l21 - l22);

    // //     // std::cout<<sum1<<' '<<sum2<<std::endl;

    // //     if ( (sum1 == 0.0) && (sum2 == 0.0)) num_intersects ++;
    // //     else {
    // //         l1T = sphericalLength(p1_sph, p2_sph);
    // //         l11 = sphericalLength(p1_sph, sph2);
    // //         l12 = sphericalLength(sph2, p2_sph);

    // //         l2T = sphericalLength(p3_sph, p4_sph);
    // //         l21 = sphericalLength(p3_sph, sph2);
    // //         l22 = sphericalLength(sph2, p4_sph);

    // //         sum1 = fabs(l1T - l11 - l12);
    // //         sum2 = fabs(l2T - l21 - l22);

    // //         if ( (sum1 == 0.0) && (sum2 == 0.0)) num_intersects ++;
    // //     }



    // //     // crossProduct(p2, c1, d2);
    // //     // crossProduct(c2, p3, d3);
    // //     // crossProduct(p4, c2, d4);

    // //     // s1 = dotProduct(d1, T);
    // //     // s2 = dotProduct(d2, T);
    // //     // s3 = dotProduct(d3, T);
    // //     // s4 = dotProduct(d4, T);

    // //     // if ( ((s1 >= 0) && (s2 >= 0) && (s3 >= 0) && (s4 >= 0))) num_intersects++;
    // //     // else if ( ((s1 < 0) && (s2 < 0) && (s3 < 0) && (s4 < 0))) num_intersects++;

    // //     // std::cout<<s1<<' '<<s2<<' '<<s3<<' '<<s4<<std::endl;

    // //     // std::cout<<v3[2]<<' '<<xyz[2]<<std::endl;
    // //     // if (isInsideSphericalTriangle(v1, v2, v3, xyz) || isOnEdge(v1, v2, xyz) ) return true;
    // }
    return false;

    // // std::cout<<"Intersect num: "<<num_intersects<<std::endl;

    // if (num_intersects%2==0) return true; // odd number, point v4 lies inside the triangle
    // else return false;
};

void Region::printCoords(void) {
    std::cout<<"REGION "<<ID<<" HAS VERTICES AT:"<<std::endl;
    for (unsigned k=0; k<triangles.size(); k++) {
        for (unsigned i=0; i<3; i++) {
            std::cout<<"   x "<<triangles[k][i][0];
        std::cout<<"   y "<<triangles[k][i][1];
        std::cout<<"   z "<<triangles[k][i][2]<<std::endl;
        }
        
    }
    std::cout<<std::endl;
    
}