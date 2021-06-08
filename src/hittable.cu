#include "hittable.cuh"

#include <glm/gtx/norm.hpp>


Sphere::Sphere() : center(glm::vec3(0.0, 0.0, 0.0)), radius(1.0f), matIdx(-1) {};
Sphere::Sphere(glm::vec3 center, float radius, int matIdx) : center(center), radius(radius), matIdx(matIdx) {};

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
  //rec.normal = (rec.point - center) / radius;
  rec.isHit = true;
  rec.matIdx = matIdx;
}

CUDA_CALLABLE_MEMBER glm::vec3 Triangle::getNormal(float u, float v) {
  return (1.0f - u - v) * v0.n + u * v1.n + v * v2.n;
}

CUDA_CALLABLE_MEMBER void Triangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) {
  glm::vec3 e0 = v1.pos - v0.pos;
  glm::vec3 e1 = v2.pos - v0.pos;
  glm::vec3 h = glm::cross(r.dir, e1);
  float a = glm::dot(e0, h);

  if (a > -0.00001 && a < 0.00001) {
    rec.isHit = false;
    return;
  }

  float f = 1.0 / a;
  glm::vec3 s = r.origin - v0.pos;
  float u = f * glm::dot(s, h);
  if (u < 0.0 || u > 1.0) {
    rec.isHit = false;
    return;
  }

  glm::vec3 q = glm::cross(s, e0);
  float v = f * glm::dot(r.dir, q);
  if (v < 0.0 || u + v > 1.0) {
    rec.isHit = false;
    return;
  }

  float t = f * glm::dot(e1, q);
  if (t > t_min && t < t_max) {
    rec.point = r.origin + t * r.dir;
    rec.t = t;
    rec.u = u;
    rec.v = v;
    //rec.setNormal(r, getNormal(u, v));
    rec.normal = getNormal(u, v);
    rec.isHit = true;
    return;
  }

  rec.isHit = false;
}