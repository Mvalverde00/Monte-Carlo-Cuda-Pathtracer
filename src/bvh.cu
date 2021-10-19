#include "bvh.cuh"

BVHNode BVH::makeLeaf(AABB& bbox, int start, int end) {
  // Negate indices to be able to tell it is a leaf.
  return BVHNode{bbox, -start, -end};
}

BVHNode BVH::makeInternal(AABB& bbox, int left, int right) {
  return BVHNode{bbox, left, right};
}

CUDA_CALLABLE_MEMBER bool BVH::traverseIterative(BVHNode* nodes, Triangle* tris, const Ray& r, float t_min, float t_max, HitRecord& hr, int root) {
  int stack[64];
  int *stackPtr = stack;
  *stackPtr++ = 0;

  HitRecord temp;
  hr.t = t_max;
  bool hit = false;
  volatile int stub = 0;

  int node = root;
  // A very small mesh might only have a leaf, so we must account
  // for that.
  if (isLeaf(nodes, node)) {
    for (int i = getStart(nodes, node); i < getEnd(nodes, node); i++) {
      if (tris[i].hit(r, t_min, t_max, temp) && temp.t < hr.t) {

        // Removing this breaks everything for some reason
        if (i == -1)
          printf("TESTING\n");

        hr = temp;
        hit = true;
      }
    }
    return hit;
  }

  do {
    int childL = getLeft(nodes, node);
    int childR = getRight(nodes, node);

    bool overlapL = getAABB(nodes, childL).hit(r, t_min, t_max);
    bool overlapR = getAABB(nodes, childR).hit(r, t_min, t_max);

    // Check for overlap against leaf nodes and add to list of
    // possible intersections if appropriate
    if (overlapL && isLeaf(nodes, childL)) {
      for (int i = getStart(nodes, childL); i < getEnd(nodes, childL); i++) {
        if (tris[i].hit(r, t_min, t_max, temp) && temp.t < hr.t) {
          if (i == -1)
            printf("TESTING\n");

          hr = temp;
          hit = true;
        }
      }
    }
    if (overlapR && isLeaf(nodes, childR)) {
      for (int i = getStart(nodes, childR); i < getEnd(nodes, childR); i++) {
        if (tris[i].hit(r, t_min, t_max, temp) && temp.t < hr.t) {
          if (i == -1)
            printf("TESTING\n");

          hr = temp;
          hit = true;
        }
      }
    }

    bool traverseL = overlapL && !isLeaf(nodes, childL);
    bool traverseR = overlapR && !isLeaf(nodes, childR);

    // If we have no active children left to traverse,
    // pop a node from the stack and start traversing it instead.
    if (!traverseL && !traverseR)
      node = *--stackPtr;
    else {
      // We have at least one active child to traverse.
      // Prioritize traversing the left child.  Push right
      // child to stack to traverse later, if applicable.
      node = traverseL ? childL : childR;
      if (traverseL && traverseR)
        *stackPtr++ = childR;
    }

  } while (node != 0);

  return hit || stub;
}


CUDA_CALLABLE_MEMBER bool BVH::isLeaf(BVHNode* nodes, int idx) {
  return getLeft(nodes, idx) < 0 || getRight(nodes, idx) < 0;
}

CUDA_CALLABLE_MEMBER int BVH::getLeft(BVHNode* nodes, int idx) {
  return nodes[idx].left;
}
CUDA_CALLABLE_MEMBER int BVH::getRight(BVHNode* nodes, int idx) {
  return nodes[idx].right;
}

CUDA_CALLABLE_MEMBER AABB& BVH::getAABB(BVHNode* nodes, int idx) {
  return nodes[idx].bbox;
};


CUDA_CALLABLE_MEMBER int BVH::getStart(BVHNode* nodes, int idx) {
  return -nodes[idx].left;
}
CUDA_CALLABLE_MEMBER int BVH::getEnd(BVHNode* nodes, int idx) {
  return -nodes[idx].right;
}