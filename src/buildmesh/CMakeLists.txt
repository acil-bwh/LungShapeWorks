cmake_minimum_required(VERSION 2.8)

project( BuildMesh )

include_directories( "../vcglib" 
                     "../PoissonRecon" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11" )

message( STATUS "CXX_FLAGS=${CMAKE_CXX_FLAGS}" )

add_executable( buildmesh trimesh_ball_pivoting.cpp )
# add_executable( buildmesh poisson_recon.cpp )

target_link_libraries( buildmesh 
                       PoissonRecon )