#include "camera.cuh"

#include <iostream>

Frustum::Frustum(float aspect, float fovy, float near) : aspect(aspect), fovy(fovy), near(near) {};
Frustum::Frustum() : aspect(16.0 / 9.0), fovy(100.0), near(1.0) {};

Camera::Camera(glm::vec3 pos, glm::mat4 rot, Frustum frustum) : pos(pos), rot(rot), frustum(frustum) {};
Camera::Camera(glm::vec3 pos, glm::mat4 rot) : pos(pos), rot(rot), frustum(Frustum()) {};
Camera::Camera(glm::vec3 pos) : pos(pos), rot(glm::mat4(1.0)), frustum(Frustum()) {};
Camera::Camera() : pos(glm::vec3(0, 0, 0)), rot(glm::mat4(1.0)), frustum(Frustum()) {};


CUDA_ONLY_MEMBER Ray Camera::getRay(int i, int j, int XRES, int YRES, curandState *d_rand) {
  float height = 2.0 * frustum.near * tan(frustum.fovy * 3.141592 / 180.0 / 2.0);
  float width = height * frustum.aspect;

  glm::vec3 e1 = glm::vec3(rot * glm::vec4(0, 0, -1, 1));
  glm::vec3 e2 = glm::vec3(rot * glm::vec4(1, 0, 0, 1));
  glm::vec3 e3 = glm::vec3(rot * glm::vec4(0, 1, 0, 1));

  float planeX = (i + curand_uniform(d_rand) - XRES / 2.0) * width / XRES;
  float planeY = (j + curand_uniform(d_rand) - YRES / 2.0) * height / YRES;

  glm::vec3 dir = frustum.near * e1 + planeX * e2 + planeY * e3;

  return Ray(pos, dir);
};


glm::mat4 Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
  glm::vec3 forward = glm::normalize(eye - center);
  glm::vec3 right = glm::cross(glm::normalize(up), forward);
  glm::vec3 newUp = glm::cross(forward, right);

  glm::mat4 out = glm::mat4();

  out[0][0] = right.x;
  out[0][1] = right.y;
  out[0][2] = right.z;
  out[1][0] = up.x;
  out[1][1] = up.y;
  out[1][2] = up.z;
  out[2][0] = forward.x;
  out[2][1] = forward.y;
  out[2][2] = forward.z;

  // No translation
  out[3][0] = 0.0;
  out[3][1] = 0.0;
  out[3][2] = 0.0;

  return out;
}
