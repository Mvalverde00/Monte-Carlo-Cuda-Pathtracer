#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "ray.h"

struct HitRecord;

class Material {
public:
  glm::vec3 color;
  float fuzzing;
  float IR;

  Material(glm::vec3 c) : color(c), fuzzing(0.0f), IR(0.0f) {};

  // returns false if no scattering
  virtual bool scatter(const Ray& in, HitRecord& rec, glm::vec3& atten, Ray& out) const = 0;

};

class Lambertian : public Material {
public:


  Lambertian(const glm::vec3& c);

  bool scatter(const Ray& in, HitRecord& rec, glm::vec3& atten, Ray& out) const;

};

#endif
