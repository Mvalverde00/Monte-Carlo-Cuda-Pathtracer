#ifndef __AABB_H__
#define __AABB_H__

#include <glm/glm.hpp>

#include "helper_cuda.h"
#include "ray.h"
#include "hittable.cuh"

class AABB {
public:
  glm::vec3 minimum;
  glm::vec3 maximum;

  AABB() : minimum(0, 0, 0), maximum(0, 0, 0) {};
  AABB(const glm::vec3& a, const glm::vec3& b) : minimum(a), maximum(b) {};
  AABB(AABB& a, AABB& b) : minimum(glm::min(a.minimum, b.minimum)), maximum(glm::max(a.maximum, b.maximum)) {};

  CUDA_CALLABLE_MEMBER bool hit(const Ray& r, float t_min, float t_max) const {
    for (int dim = 0; dim < 3; dim++) {
      float invD = 1.0f / r.dir[dim];
      float t0 = (minimum[dim] - r.origin[dim]) * invD;
      float t1 = (maximum[dim] - r.origin[dim]) * invD;

      // swap
      if (invD < 0.0f) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
      }

      t_min = t0 > t_min ? t0 : t_min;
      t_max = t1 < t_max ? t1 : t_max;
      if (t_max < t_min) {
        return false;
      }
    }
    return true;
  };

  // surface area.  For a point-like bounding box
  // (i.e. with 0 volume), we define the surface area to be infinity.
  inline float area() {
    if (minimum == maximum) {
      //std::cout << "Creating infinite area\n";
      return 999999999999.0f; // Infinity
    }

    float s1 = maximum.x - minimum.x;
    float s2 = maximum.y - minimum.y;
    float s3 = maximum.z - minimum.z;
    return 2.f * (s1 * s2 + s2 * s3 + s3 * s1);
  }
};


inline bool box_compare(const Triangle& a, const Triangle& b, int axis) {
  AABB box_a, box_b;
  a.bbox(box_a);
  b.bbox(box_b);

  return box_a.minimum[axis] < box_b.minimum[axis];
}

inline bool box_compare_x(const Triangle& a, const Triangle& b) { return box_compare(a, b, 0); }
inline bool box_compare_y(const Triangle& a, const Triangle& b) { return box_compare(a, b, 1); }
inline bool box_compare_z(const Triangle& a, const Triangle& b) { return box_compare(a, b, 2); }


inline bool box_compare_centroid(const Triangle& a, const Triangle& b, int axis) {
  AABB box_a, box_b;
  a.bbox(box_a);
  b.bbox(box_b);

  // Can skip multiplying both sides by 0.5f since it does not affect inequality.
  return (box_a.minimum + box_a.maximum)[axis] < (box_b.minimum + box_b.maximum)[axis];
}

inline bool box_compare_centroid_x(const Triangle& a, const Triangle& b) { return box_compare_centroid(a, b, 0); }
inline bool box_compare_centroid_y(const Triangle& a, const Triangle& b) { return box_compare_centroid(a, b, 1); }
inline bool box_compare_centroid_z(const Triangle& a, const Triangle& b) { return box_compare_centroid(a, b, 2); }

#endif