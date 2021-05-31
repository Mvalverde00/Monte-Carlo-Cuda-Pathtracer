#include "controls.h"

#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include "helper_cuda.h"

glm::vec3 vel;
float yaw = 0.0f;
float pitch = 0.0f;

void updateCamera(Keyboard& keyboard, Mouse& mouse, Camera* cam, PTData& data, float dt) {
  vel = glm::vec3(0, 0, 0);
  if (keyboard.isKeyDown(SDLK_w)) {
    vel.z -= 1.0;
  }
  if (keyboard.isKeyDown(SDLK_s)) {
    vel.z += 1.0;
  }
  if (keyboard.isKeyDown(SDLK_a)) {
    vel.x -= 1.0;
  }
  if (keyboard.isKeyDown(SDLK_d)) {
    vel.x += 1.0;
  }
  if (keyboard.isKeyDown(SDLK_SPACE)) {
    vel.y += 1.0;
  }
  if (keyboard.isKeyDown(SDLK_LSHIFT)) {
    vel.y -= 1.0;
  }
  float norm = glm::length(vel);
  if (norm != 0.0) {
    vel = 1.2f * vel / norm;
  }

  bool rotChange = false;
  if (mouse.isPressed(SDL_BUTTON_RIGHT) && mouse.getDx() + mouse.getDy() != 0) {
    rotChange = true;

    yaw -= mouse.getDx();
    pitch -= mouse.getDy();
    pitch = glm::clamp(pitch, -89.5f, 89.5f); // Constrain up/down rotation
  }

  glm::quat yawRot = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(yaw), glm::vec3(0, 1, 0));
  glm::quat rot = yawRot * glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(pitch), glm::vec3(1, 0, 0));
  cam->rot = glm::mat4_cast(rot);

  vel = glm::vec3(yawRot * glm::vec4(vel, 1.0));
  cam->pos += vel * dt;

  // There was a change, meaning the camera should be updated and framebuffer reset
  if (norm != 0.0 || rotChange) {
    data.reset = true;
    data.samples = 0;

    CUDA_CALL(cudaMemcpy(data.cam, cam, sizeof(Camera), cudaMemcpyHostToDevice));
  }
}


void displayCameraStats(Camera* cam, PTData& data) {
  ImGui::Begin("Rendering Info");
  ImGui::Text("Camera Pos: %f %f %f", cam->pos.x, cam->pos.y, cam->pos.z);
  ImGui::Text("Camera yaw/pitch: %f %f", yaw, pitch);
  ImGui::Text("Samples accumulated: %d", data.samples);
}