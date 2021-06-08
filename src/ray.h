#ifndef __RAY_H__
#define __RAY_H__

#include <glm/glm.hpp>

#include "helper_cuda.h"

class Ray {
public:
  glm::vec3 origin;
  //float padding1;
  glm::vec3 dir;
  //float padding2;


  CUDA_CALLABLE_MEMBER Ray() : origin(glm::vec3(0.0, 0.0, 0.0)), dir(glm::vec3(0.0, 0.0, 1.0)) {};
  CUDA_CALLABLE_MEMBER Ray(glm::vec3 origin, glm::vec3 dir) : origin(origin), dir(dir) {};
  
  CUDA_CALLABLE_MEMBER glm::vec3 at(float t) const {
    return origin + t * dir;
  };
};

#endif