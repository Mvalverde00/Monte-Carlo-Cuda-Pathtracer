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

  int matIdx; // index of material of hit object

  // Used for texturing the triangle and interpolating normal.
  float u;
  float v;


  inline void setNormal(const Ray& r, const glm::vec3& n) {
    front = glm::dot(r.dir, n) < 0.0f;
    normal = front ? n : -n;
  }
};

class Sphere {
public:
  glm::vec3 center;
  float radius;
  int matIdx;

  Sphere();
  Sphere(glm::vec3 center, float radius, int matIdx);

  CUDA_CALLABLE_MEMBER void hit(const Ray& r, float t_min, float t_max, HitRecord& rec);
};


struct Vertex {
  glm::vec3 pos;
  glm::vec3 n;

  Vertex(glm::vec3 pos, glm::vec3 n) : pos(pos), n(n) {};
};

/* A triangle which is part of a mesh
 * No need to store own material, uses same as container mesh.
 * Not technically hittable on its own, since it needs info from container mesh
 * to be complete. */
class Triangle {
public:
  Vertex v0, v1, v2;

  Triangle(Vertex v0, Vertex v1, Vertex v2) : v0(v0), v1(v1), v2(v2) {};

  // Given barycentric coords u and v, with w = 1 - u - v, return the interpolated
  // normal at those coordinates
  CUDA_CALLABLE_MEMBER glm::vec3 getNormal(float u, float v);

  CUDA_CALLABLE_MEMBER void hit(const Ray& r, float t_min, float t_max, HitRecord& rec);
};

#endif