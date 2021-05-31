#include "material.cuh"

#include "hittable.cuh"



Lambertian::Lambertian(const glm::vec3& c) : Material(c) {};

bool Lambertian::scatter(const Ray& in, HitRecord& rec, glm::vec3& atten, Ray& out) const {
  atten = color;
  /*
  glm::vec3 outDir = rec.normal + randUnitVec();
  if (nearZero(outDir))
    outDir = rec.normal;
  out = Ray(rec.point, outDir);*/
  return true;
}