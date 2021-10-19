#ifndef __BVH_H__
#define __BVH_H__

#include <vector>

#include "aabb.cuh"
#include "ray.h"
#include "hittable.cuh"

// Forward Declaration
class Mesh;


// 32-byte aligned
struct BVHNode {
  AABB bbox; // 6 floats = 24 bytes

  /* when positive, left/right store the indices of the
   * corresponding left/right BVHnodes.  When negative,
   * the BVHNode is a leaf node and the interval 
   * [-left, -right) are the triangles contained. */
  int left; // 4 bytes
  int right; // 4 bytes

};

namespace BVH {
  // Create a leaf node corresponding to triangle indices in [start, end)
  BVHNode makeLeaf(AABB& bbox, int start, int end);

  // Create an internal node with the corresponding left and right nodes
  BVHNode makeInternal(AABB& bbox, int left, int right);

  // Abstract away common actions since BVH will likely undergo significant changes
  CUDA_CALLABLE_MEMBER bool isLeaf(BVHNode* nodes, int idx);
  CUDA_CALLABLE_MEMBER int getLeft(BVHNode* nodes, int idx);
  CUDA_CALLABLE_MEMBER int getRight(BVHNode* nodes, int idx);
  CUDA_CALLABLE_MEMBER int getStart(BVHNode* nodes, int idx);
  CUDA_CALLABLE_MEMBER int getEnd(BVHNode* nodes, int idx);
  CUDA_CALLABLE_MEMBER AABB& getAABB(BVHNode* nodes, int idx);

  // BVH traversal using above abstractions
  CUDA_CALLABLE_MEMBER bool traverseIterative(BVHNode* nodes, Triangle* tris, const Ray& r, float t_min, float t_max, HitRecord& hr, int root);
};

#endif