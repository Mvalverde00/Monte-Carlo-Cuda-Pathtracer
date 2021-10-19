#ifndef __MESH_H__
#define __MESH_H__

#include "hittable.cuh"
#include "material.cuh"
#include "bvh.cuh"

#include <vector>
#include <string>

/* Contains data defining a triangle mesh.
 * Not hittable itself, but rather used as a base for a
 * specific MeshInstance-- i.e. a version of the mesh
 * with a specific material and transformations applied.
 * Purely for bookkeeping on the CPU, not ever send to GPU. */
class Mesh {
  // helper method for main public initAcceleration function.  Returns
  // index of created root bvh node.
  int initAcceleration(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris, int start, int end);

public:
  // offset of this mesh's triangles in global triangle list
  int triOffset;
  // number of tris in the global triangle list.  Thus, the triangles
  // in the mesh are [triOffset, triOffset + n_tris - 1]
  int n_tris;
  // offset of this mesh's root node in global BVH node list
  int bvhOffset;

  // Create a mesh by loading in all of the triangles
  // from the .obj file located at fpath, and add the results
  // to the global vector of triangles.
  Mesh(std::string fpath, std::vector<Triangle>& tris);

  Mesh() : triOffset(-1), n_tris(-1), bvhOffset(-1) {};

  /* After the mesh has been fully loaded, calling this function will build a BVH
  * acceleration structure and append it to the input vector */
  void initAcceleration(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris);
};



class MeshInstance {
public:
  // offset of this mesh's triangles in global triangle list
  int triOffset;
  // number of tris in the global triangle list.  Thus, the triangles
  // in the mesh are [triOffset, triOffset + n_tris - 1]
  int n_tris;

  // offset of this mesh's root node in global BVH node list
  int bvhOffset;

  int matIdx;
  glm::mat4 tMat; // Object to world matrix
  glm::mat4 tMatInv; // World to object matrix
  glm::mat4 tMatInvTpose; // World to object for normals

  MeshInstance(Mesh& base, int matIdx, glm::mat4 tMat);

  CUDA_CALLABLE_MEMBER bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, BVHNode* nodes, Triangle* tris);
};

#endif