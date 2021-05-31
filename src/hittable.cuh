#ifndef __HITTABLE_H__
#define __HITTABLE_H__

#include <glm/glm.hpp>

#include "ray.h"
#include "helper_cuda.h"


struct HitRecord {
  // Whether or not there was a valid hit.  If false, should not attempt to
  // access any other data fields as they will be meaningless.
  bool isHit;

  glm::vec3 point;
  glm::vec3 normal; // Normal must always point against the incident ray.
  float t; // the parameter at which the ray made the hit.

  bool front; // Did we hit front of the face?

  //Material* mat;


  inline void setNormal(const Ray& r, const glm::vec3& n) {
    front = glm::dot(r.dir, n) < 0.0;
    normal = front ? n : -n;
  }
};

class Sphere {
public:
  glm::vec3 center;
  float radius;
  //Material* mat;

  Sphere();
  Sphere(glm::vec3 center, float radius);

  CUDA_CALLABLE_MEMBER void hit(const Ray& r, float t_min, float t_max, HitRecord& rec);
};

#endif