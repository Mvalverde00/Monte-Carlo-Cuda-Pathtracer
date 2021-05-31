#include "hittable.cuh"

#include <glm/gtx/norm.hpp>


Sphere::Sphere() : center(glm::vec3(0.0, 0.0, 0.0)), radius(1.0f) {};
Sphere::Sphere(glm::vec3 center, float radius) : center(center), radius(radius) {};

CUDA_CALLABLE_MEMBER void Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) {
  // Math taken from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

  glm::vec3 oc = r.origin - center;

  float a = glm::length2(r.dir);
  float b = 2.0 * glm::dot(r.dir, oc);
  float c = glm::length2(oc) - radius * radius;

  float discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    rec.isHit = false;
    return;
  }
  float sqrtd = std::sqrt(discriminant);

  float root = (-b - sqrtd) / (2.0 * a);
  if (root < t_min || root > t_max) {
    root = (-b + sqrtd) / (2.0 * a);
    if (root < t_min || root > t_max) {
      rec.isHit = false;
      return;
    }
  }

  rec.t = root;
  rec.point = r.at(root);
  rec.setNormal(r, (rec.point - center) / radius);
  rec.isHit = true;
  //rec.mat = mat;
}