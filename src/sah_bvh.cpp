#include "sah_bvh.h"

#include <cassert>
#include <algorithm>

#include "aabb.cuh"


namespace SAH_BVH {

  const float C_tri = 1.0f;
  const float C_step = 0.0f;

  inline float SAH_Heuristic(int n_s1, float area_s1, int n_s2, float area_s2, float area_total) {
    return (2.f * C_step) + (area_s1 / area_total * n_s1 * C_tri) + (area_s2 / area_total * n_s2 * C_tri);
  }

  inline float SAH_Heuristic2(int n_s1, float area_s1, int n_s2, float area_s2) {
    // Since 2.0f * C_step is a constant, we can simply minimize the rest of the equation.
    return area_s1 * n_s1 + area_s2 * n_s2;
  }

  int createBVH(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris, int start, int end, int depth) {
    //std::cout << "depth " << depth << ", span " << end - start << "\n";
    int span = end - start;
    float bestCost = C_tri * span;
    int bestAxis = -1;
    int bestEvent = -1;

    for (int axis = 0; axis < 3; axis++) {
      auto comparator = axis == 0 ? box_compare_centroid_x
        : axis == 1 ? box_compare_centroid_y
        : box_compare_centroid_z;

      std::sort(tris.begin() + start, tris.begin() + end, comparator);

      std::vector<float> leftAreas(span);
      AABB totalBBox, triBBox;
      totalBBox = AABB();
      for (int i = start; i < end; i++) {
        leftAreas[i - start] = totalBBox.area();
        
        // Expand bbox to include tri
        tris[i].bbox(triBBox);
        if (i == start)
          totalBBox = triBBox;
        else
          totalBBox = AABB(totalBBox, triBBox);

      }
      float totalArea = totalBBox.area();

      std::vector<float> rightAreas(span);
      totalBBox = AABB(); // reset overall bbox
      for (int i = end - 1; i >= start; i--) {
        rightAreas[i - start] = totalBBox.area();

        // Expand bbox to include tri
        tris[i].bbox(triBBox);
        if (i == end - 1)
          totalBBox = triBBox;
        else
          totalBBox = AABB(totalBBox, triBBox);

        float thisCost = SAH_Heuristic(i - start + 1, leftAreas[i - start], span - i + start - 1, rightAreas[i - start], totalArea);

        if (thisCost < bestCost) {
          bestCost = thisCost;
          bestEvent = i;
          bestAxis = axis;
        }
      }

    }

    if (bestAxis == -1) {
      // Best option is to just make current selection a leaf node
      AABB triBBox;
      tris[start].bbox(triBBox);
      AABB total = triBBox;
      for (int i = start; i < end; i++) {
        tris[i].bbox(triBBox);
        total = AABB(total, triBBox);
      }

      nodes.push_back(BVH::makeLeaf(total, start, end));
      return int(nodes.size()) - 1;
    } else {
      auto comparator = bestAxis == 0 ? box_compare_centroid_x
        : bestAxis == 1 ? box_compare_centroid_y
        : box_compare_centroid_z;

      std::sort(tris.begin() + start, tris.begin() + end, comparator);

      // reserve space for current internal node
      nodes.push_back(BVHNode());
      int idx = int(nodes.size()) - 1;

      int left = createBVH(nodes, tris, start, bestEvent, depth + 1);
      int right = createBVH(nodes, tris, bestEvent, end, depth + 1);

      // Update internal node with correct data
      AABB merged = AABB(BVH::getAABB(&nodes[0], left), BVH::getAABB(&nodes[0], right));
      nodes[idx] = BVH::makeInternal(merged, left, right);

      //std::cout << "axis " << bestAxis << " event " << bestEvent - start << "\n";
      return idx;
    }
  }

}