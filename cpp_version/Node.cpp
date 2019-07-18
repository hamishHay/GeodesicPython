#include "Node.h"
#include "math_functions.h"
#include <iostream>
#include <algorithm>
#include <vector>

Node::Node(double xyz[], int ID_num)
{
  xyz_coords[0] = xyz[0];
  xyz_coords[1] = xyz[1];
  xyz_coords[2] = xyz[2];
  ID = ID_num;

  cart2sph(xyz_coords, sph_coords);
  // friends_list
};

Node::Node(const Node &other_node)
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

void Node::transformSph(const double rot)
{
    double x = this->xyz_coords[0];
    double y = this->xyz_coords[1];

    this->xyz_coords[0] = cos(rot)*x + sin(rot)*y;
    this->xyz_coords[1] = -sin(rot)*x + cos(rot)*y;

    cart2sph(xyz_coords, sph_coords);
}

double Node::getMagnitude()
{
    double mag;

    mag = sqrt(this->xyz_coords[0]*this->xyz_coords[0]
              +this->xyz_coords[1]*this->xyz_coords[1]
              +this->xyz_coords[2]*this->xyz_coords[2]);

    return mag;
}

void Node::project2Sphere(double r)
{
    double mag;

    mag = this->getMagnitude();

    this->xyz_coords[0] /= mag;
    this->xyz_coords[1] /= mag;
    this->xyz_coords[2] /= mag;

    // Maybe we shouldn't do this here?
    cart2sph(xyz_coords, sph_coords);
}

void Node::addFriend(Node * n)
{
    friends_list.push_back(n);
}

int Node::isFriend(Node * n)
{
    int i;
    for (i=0; i<friends_list.size(); i++)
    {
        if (*n == *friends_list[i]) return 1;
    }
    return 0;
}

void Node::addTempFriend(Node * n)
{
    temp_friends.push_back(n);
}

void Node::orderFriends()
{
  struct Pair
  {
      int ID1, ID2;
      Node * n1, * n2;
      Pair(int i1, int i2, Node * in1, Node * in2) : ID1(i1), ID2(i2), n1(in1), n2(in2) { };

      void reversePair()
      {
        int IDt = ID1;
        Node * nt = n1;
        ID1 = ID2;
        ID2 = IDt;
        n1 = n2;
        n2 = nt;
      }

      int isEqual(Pair x)
      {
          if ((ID1 == x.ID1 || ID1 == x.ID2) && (ID2 == x.ID1 || ID2 == x.ID2))
            {return 1;}
          else
            {return 0;}
      }

  };

  int i,j;
  int pent = 0;
  int dead = 0;
  Node * node;
  bool ordered = false;
  std::vector<Node *> alive_list;
  std::vector<Node *> dead_list;
  std::vector<Node *> ordered_list;
  std::vector<Pair> pairs;
  std::vector<Pair> ordered_pairs;
  if (friend_num==5) pent = 1;


  for (i=0; i<friends_list.size()-pent; i++)
  {
    if (friends_list[i]->boundary == -1) {
        dead += 1;
        dead_list.push_back(friends_list[i]);
    }
    else alive_list.push_back(friends_list[i]);
  }

  this->dead_num = dead;

    std::vector<double> dist(alive_list.size());
    for (i=0; i<alive_list.size(); i++)
    {
      for (j=0; j<alive_list.size(); j++)
      {
        if (i!=j) dist[j] = (*alive_list[i] - *alive_list[j]).getMagnitude();
        else      dist[j] = 10.0;
        // std::cout<<alive_list[i]->ID<<": "<<alive_list[j]->ID<<": "<<dist[j]<<"     ";
      }
      // std::cout<<std::endl;

      // std::vector<int>::iterator result = std::min_element(dist.begin(), dist.end());
      // int indx = std::distance(dist.begin(), result);
      int indx = std::distance( dist.begin(), std::min_element(dist.begin(),dist.end()));

      // int indx = std::distance(dist, std::min_element(dist, dist + alive_list.size()-1));
      // std::cout<<ID<<" PAIR "<<alive_list[i]->ID<<' '<<alive_list[indx]->ID<<std::endl;

      if (alive_list[i]->ID != alive_list[indx]->ID)
        pairs.push_back( Pair( alive_list[i]->ID, alive_list[indx]->ID, alive_list[i], alive_list[indx]) );
    }

    for (i=0; i<pairs.size(); i++)
    {
        for (j=0; j<pairs.size(); j++)
        {
            if (i!=j) {
                if (pairs[i].isEqual(pairs[j])) {
                    pairs.erase( pairs.begin() + j );
                }
            }
        }
    }

    int NP = pairs.size();
    // std::cout<<"NO PAIRS: "<<NP<<std::endl;
    ordered_pairs.push_back(pairs[0]);
    pairs.erase( pairs.begin() + 0 );
    if (NP > 1) {
        int iter = 0;
        while (ordered_pairs.size() < NP) {
            bool added = false;

            for (i=0; i<2; i++) {
                int ii=0;
                if (i==1) ii = ordered_pairs.size()-1;

                for (j=0; j<pairs.size(); j++) {
                    // std::cout<<ID<<' '<<pairs[j].ID1<<' ';
                    // std::cout<<pairs[j].ID2<<' ';
                    // std::cout<<ordered_pairs[ii].ID1<<' ';
                    // std::cout<<ordered_pairs[ii].ID2<<std::endl;

                    if (pairs[j].ID1 == ordered_pairs[ii].ID2) {
                        ordered_pairs.push_back(pairs[j]);
                        pairs.erase( pairs.begin() + j );
                        added = true;
                    }
                    else if (pairs[j].ID2 == ordered_pairs[ii].ID2) {
                        pairs[j].reversePair();
                        ordered_pairs.push_back(pairs[j]);
                        pairs.erase( pairs.begin() + j );
                        added = true;
                    }
                    else if (pairs[j].ID1 == ordered_pairs[ii].ID1) {
                        pairs[j].reversePair();
                        ordered_pairs.insert(ordered_pairs.begin() + 0, pairs[j]);
                        pairs.erase( pairs.begin() + j );
                        added = true;
                    }
                    else if (pairs[j].ID2 == ordered_pairs[ii].ID1) {
                        ordered_pairs.insert(ordered_pairs.begin() + 0, pairs[j]);
                        pairs.erase( pairs.begin() + j );
                        added = true;
                    }
                }
            }

            iter++;

            if (iter>10)
            {
                Node * n1, * n2;
                n1 = ordered_pairs[0].n1;
                n2 = ordered_pairs[ordered_pairs.size()-1].n2;

                for (i=0; i<pairs.size(); i++)
                {
                    if (n1->isFriend(pairs[i].n1)) {
                        pairs.push_back( Pair( n1->ID, pairs[i].n1->ID, n1, pairs[i].n1) );
                        NP++;
                        break;
                    }
                    else if (n1->isFriend(pairs[i].n2)) {
                        pairs.push_back( Pair( n1->ID, pairs[i].n2->ID, n1, pairs[i].n2) );
                        NP++;
                        break;
                    }
                    else if (n2->isFriend(pairs[i].n1)) {
                        pairs.push_back( Pair( n2->ID, pairs[i].n1->ID, n2, pairs[i].n1) );
                        NP++;
                        break;
                    }
                    else if (n2->isFriend(pairs[i].n2)) {
                        pairs.push_back( Pair( n2->ID, pairs[i].n2->ID, n2, pairs[i].n2) );
                        NP++;
                        break;
                    }
                }
                iter = 0;
            }
        }
    }
        // std::cout<<"ordered: ";
        int count = 0;
        // std::cout<<"ID: "<<ID<<std::endl;
        if (ordered_pairs.size() == 1)
        {
            friends_list[0] = ordered_pairs[0].n1;
            friends_list[1] = ordered_pairs[0].n2;
            count += 2;
            // std::cout<<ordered_pairs[0].ID1<<' '<<ordered_pairs[0].ID2<<' ';
        }
        else
        {
            for (i=0; i<ordered_pairs.size(); i++)
            {
                // std::cout<<ordered_pairs[i].ID1<<' '<<ordered_pairs[i].ID2<<' ';

                friends_list[i] = ordered_pairs[i].n1;
                count++;
            }
            friends_list[i] = ordered_pairs[i-1].n2;
            count++;
        }


        for (i=0; i<dead; i++) {
            // double xyz[3] = {-1, -1, -1};
            // Node * n1 = new Node(xyz, -2);
            // dead_list[i]->ID = -2;

            friends_list[count++] = dead_list[i];
            // friends_list[count++] = n1;
        }
        if (pent) {
            double xyz[3] = {-1, -1, -1};
            Node * n1 = new Node(xyz, -1);
            friends_list[5] = n1;
        }

        // std::cout<<std::endl;
        // for (i=0; i<friend_num; i++) std::cout<<friends_list[i]->ID<<' ';

        // std::cout<<std::endl<<ordered_pairs.size()-1<<' '<<i<<std::endl;


}

void Node::printCoords()
{
    // std::cout<<"Coordinates of node "<<this->ID<<": "<<std::endl;
    // std::cout<<"x: "<<this->xyz_coords[0]<<", ";
    // std::cout<<"y: "<<this->xyz_coords[1]<<", ";
    // std::cout<<"z: "<<this->xyz_coords[2]<<std::endl;

    std::cout<<"Coordinates of node "<<this->ID<<": "<<std::endl;
    std::cout<<"lat: "<<this->sph_coords[1]*180./pi<<", ";
    std::cout<<"lon: "<<this->sph_coords[2]*180./pi<<std::endl;
}

void Node::getMapCoords(const Node &center_node, double xy[])
{
  double m;
  double lat1, lat2, lon1, lon2;

  lat1 = center_node.sph_coords[1];
  lon1 = center_node.sph_coords[2];
  lat2 = this->sph_coords[1];
  lon2 = this->sph_coords[2];

  m = 2.0 / (1.0 + sin(lat2)*sin(lat1) + cos(lat1)*cos(lat2)*cos(lon2-lon1));

  xy[0] = m * cos(lat2) * sin(lon2 - lon1);
  xy[1] = m * (sin(lat2)*cos(lat1) - cos(lat2)*sin(lat1)*cos(lon2-lon1));
}

void Node::updateXYZ(const double xyz[])
{
    this->xyz_coords[0] = xyz[0];
    this->xyz_coords[1] = xyz[1];
    this->xyz_coords[2] = xyz[2];

    cart2sph(xyz_coords, sph_coords);
}

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
