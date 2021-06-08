#include "material.cuh"

#include "hittable.cuh"
#include <curand_kernel.h>
#include <glm/gtx/norm.hpp>

// Random vector on the unit sphere
CUDA_ONLY_MEMBER inline glm::vec3 randUnitVec(curandState* rand) {
  return glm::normalize(glm::vec3(curand_uniform(rand) * 2.0f - 1.0f, curand_uniform(rand) * 2.0f - 1.0f, curand_uniform(rand) * 2.0f - 1.0f));
}

// Random vector inside unit sphere
CUDA_ONLY_MEMBER inline glm::vec3 randPointSphere(curandState* rand) {
  // https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere

  return randUnitVec(rand) * (powf(curand_uniform(rand), 1.0f / 3.0f));
}

CUDA_ONLY_MEMBER inline bool nearZero(glm::vec3& v) {
  return fabs(v.x) < 0.00001f && fabs(v.y) < 0.00001f && fabs(v.z) < 0.00001f;
}
CUDA_ONLY_MEMBER inline float schlick(float cosTheta, float eta) {
  float r0 = (1.0f - eta) / (1.0f + eta);
  r0 = r0 * r0;

  return r0 + (1.0f - r0) * powf((1.0f - cosTheta), 5);
}

CUDA_ONLY_MEMBER inline glm::vec3 refract(const glm::vec3& in, const glm::vec3& n, float eta, float cosTheta) {
  glm::vec3 out_perp = eta * (in + cosTheta * n);
  glm::vec3 out_par = -sqrt(abs(1.0f - glm::length2(out_perp))) * n;
  return out_perp + out_par;
}

// Need to have some base method, although in practice this should never be used.
CUDA_ONLY_MEMBER bool Material::scatter(Ray& in, HitRecord& rec, glm::vec3& atten, curandState* rand) const {
  switch (type) {
    case MaterialType::LAMBERTIAN :
      atten = color;
      glm::vec3 outDir = rec.normal + randUnitVec(rand);
      if (nearZero(outDir))
        outDir = rec.normal;
      in = Ray(rec.point, outDir);
      return true;
    case MaterialType::METAL :
      atten = color;
      in = Ray(rec.point, glm::reflect(glm::normalize(in.dir), rec.normal));
      in.dir += fuzzing * randPointSphere(rand);

      return glm::dot(in.dir, rec.normal) > 0.0f;
    case MaterialType::DIALECTRIC :
      atten = color;
      //bool front = glm::dot(rec.normal, in.dir) < 0.0f;
      float eta = rec.front ? (1.0 / IR) : IR;

      glm::vec3 inDir = glm::normalize(in.dir);
      //rec.normal = front ? rec.normal : -rec.normal;
      float cosTheta = fmin(1.0f, glm::dot(-inDir, rec.normal));
      float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
      
      if ((eta * sinTheta > 1.0f) || (schlick(cosTheta, eta) > curand_uniform(rand)))
        outDir = glm::reflect(inDir, rec.normal);
      else
        outDir = refract(inDir, rec.normal, eta, cosTheta);

      in = Ray(rec.point, outDir);
      return true;

    default:
      return false;

  }
}

CUDA_ONLY_MEMBER glm::vec3 Material::emit() const {
  switch (type) {
    case MaterialType::LIGHT:
      return color;
    default:
      return glm::vec3(0, 0, 0);
  }
}

Material makeLambertian(glm::vec3 color) {
  return Material(color);
}

Material makeMetal(glm::vec3 color) {
  Material mat = Material();
  mat.color = color;
  mat.fuzzing = 0.0f;
  mat.type = MaterialType::METAL;

  return mat;
}
Material makeMetal(glm::vec3 color, float fuzz) {
  Material mat = Material();
  mat.color = color;
  mat.fuzzing = fuzz;
  mat.type = MaterialType::METAL;

  return mat;
}

Material makeDialectric(float IR) {
  Material mat = Material();
  mat.color = glm::vec3(1,1,1);
  mat.IR = IR;
  mat.type = MaterialType::DIALECTRIC;

  return mat;
}

Material makeLight(glm::vec3 emitted) {
  Material mat = Material();
  mat.color = emitted;
  mat.type = MaterialType::LIGHT;

  return mat;
}