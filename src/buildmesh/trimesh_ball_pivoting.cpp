/****************************************************************************
* VCGLib                                                            o o     *
* Visual and Computer Graphics Library                            o     o   *
*                                                                _   O  _   *
* Copyright(C) 2004-2012                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/
#include <vcg/complex/complex.h>

#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/complex/algorithms/create/ball_pivoting.h>

// input output
#include <wrap/io_trimesh/import_asc.h>
#include <wrap/io_trimesh/export_obj.h>

using namespace vcg;
using namespace std;

class MyFace;
class MyVertex;

struct MyUsedTypes : public UsedTypes< Use<MyVertex>::AsVertexType, Use<MyFace>::AsFaceType > {};

class MyVertex  : public Vertex< MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags, vertex::Mark>{};
class MyFace    : public Face  < MyUsedTypes, face::VertexRef, face::Normal3f, face::BitFlags > {};
class MyMesh    : public vcg::tri::TriMesh< vector<MyVertex>, vector<MyFace> > {};

bool callback(int percent, const char *str)
{
  //cout << "str: " << str << " " << percent << "%\n";
  return true;
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    printf(
      "Usage: %s filein.asc fileout.obj [opt]\n"
      "options: \n"
      "-r <val> radius of the rolling ball\n"
      "-c <val> clustering radius (as fraction of radius). default: 0.2\n"
      "-a <val> max angle between two faces. default: 90 degrees\n", argv[0]
    );
    exit(0);
  }

  float radius = 0.0f;
  float clustering = 0.2;
  float angle = 90;
  int i = 3;
  while (i < argc)
  {
    if (argv[i][0]!='-')
    {
      printf("Error unable to parse option '%s'\n",argv[i]); exit(0);
    }
    switch(argv[i][1])
    {
        case 'r':
          radius = atof(argv[++i]);
          printf("Using %f sphere radius\n", radius);
          break;
        case 'c':
          clustering = atof(argv[++i]);
          printf("Using %f clustering radius\n", clustering);
          break;
        case 'a':
          angle = atof(argv[++i]);
          printf("Using %f maximum angle (degree)\n", angle);
          break;
        default:
          printf("Error unable to parse option '%s'\n",argv[i]);
          exit(0);
    }
    ++i;
  }

  if (radius == 0)
  {
    printf("Autodetecting ball radius...\n");
  }

  MyMesh m;

  if (vcg::tri::io::ImporterASC<MyMesh>::Open(m,argv[1])!=0)
  {
    printf("Error reading file  %s\n",argv[1]);
    exit(0);
  }

  vcg::tri::UpdateBounding<MyMesh>::Box(m);
  vcg::tri::UpdateNormal<MyMesh>::PerFace(m);
  printf("Input mesh  vn:%i fn:%i\n",m.VN(),m.FN());

  // Initialization
  float angle_radian = angle * M_PI / 180.0;
  tri::BallPivoting<MyMesh> pivot(m, radius, clustering, angle_radian);
  printf("Ball radius: %f\nClustering points within %f radii\n", pivot.radius, clustering);

  // the main processing
  pivot.BuildMesh(callback);

  printf("Output mesh vn:%i fn:%i\n",m.VN(),m.FN());

  int mask = 0;
  vcg::tri::io::ExporterOBJ<MyMesh>::Save(m,argv[2],mask);
  return 0;
}
