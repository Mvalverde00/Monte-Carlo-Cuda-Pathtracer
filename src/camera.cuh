#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <glm/glm.hpp>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "ray.h"

struct Frustum {
  float aspect;
  float fovy; // Stored in degrees
  float near;

  Frustum(float aspect, float fovy, float near);
  Frustum();
};

class Camera {
public:
  glm::vec3 pos;
  glm::mat4 rot;
  Frustum frustum;

  Camera(glm::vec3 pos, glm::mat4 rot, Frustum frustum);
  Camera(glm::vec3 pos, glm::mat4 rot);
  Camera(glm::vec3 pos);
  Camera();

  CUDA_ONLY_MEMBER Ray getRay(int i, int j, int XRES, int YRES, curandState* d_rand);

  static glm::mat4 lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up);

};

#endif
