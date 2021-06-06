#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "ray.h"
#include "helper_cuda.h"

#include <curand_kernel.h>

struct HitRecord;

enum class MaterialType {
  LAMBERTIAN,
  METAL,
  DIALECTRIC
};

class Material {
public:
  glm::vec3 color;
  float fuzzing;
  float IR;
  MaterialType type;

  Material() {};
  Material(glm::vec3 c) : color(c), fuzzing(0.0f), IR(0.0f), type(MaterialType::LAMBERTIAN) {};

  // returns false if no scattering
  // Overwrites the "in" ray with the new output ray.
  CUDA_ONLY_MEMBER bool scatter(Ray& in, HitRecord& rec, glm::vec3& atten, curandState* rand) const;

};

Material makeLambertian(glm::vec3 color);
Material makeMetal(glm::vec3 color);
Material makeMetal(glm::vec3 color, float fuzz);
Material makeDialectric(float IR);

#endif
