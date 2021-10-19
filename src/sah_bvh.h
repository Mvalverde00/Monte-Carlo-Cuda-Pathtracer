#ifndef __SAH_BVH_H__
#define __SAH_BVH_H__

#include <vector>

#include "bvh.cuh"

/* Implemenation of Surface Area Heuristic BVH taken from
 * https://graphics.stanford.edu/~boulos/papers/togbvh.pdf */
namespace SAH_BVH {
  // Creats a BVH encapsulating the triangles located in the tris array with indices in [start, end)
  // Populates the corresponding BVH nodes into the input "nodes" array.  Returns an int representing the
  // index of the created node in the "nodes" array.
  int createBVH(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris, int start, int end, int depth = 0);
}



#endif