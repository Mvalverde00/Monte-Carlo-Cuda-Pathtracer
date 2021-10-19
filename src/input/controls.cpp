#include "controls.h"

#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include "helper_cuda.h"
#include "scene.h"

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

    if (keyboard.isKeyDown(SDLK_r)) {
      vel *= 80.0f;
    }
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
    data.renderTime = 0.0f;

    CUDA_CALL(cudaMemcpy(data.cam, cam, sizeof(Camera), cudaMemcpyHostToDevice));
  }
}


void displayCameraStats(Camera* cam, PTData& data) {
  ImGui::Begin("Rendering Info");
  ImGui::Text("Camera Pos: %f %f %f", cam->pos.x, cam->pos.y, cam->pos.z);
  ImGui::Text("Camera yaw/pitch: %f %f", yaw, pitch);
  ImGui::Text("Samples accumulated: %d", data.samples);
  ImGui::Text("Total render time: %f", data.renderTime);
  ImGui::Text("Average Samples per Second: %f", data.samples / data.renderTime);
  ImGui::Text("Geometry: Rendering %d spheres and %d meshes.", data.n_sphs, data.n_meshes);

  if (ImGui::Checkbox("Show object normals", &data.showNormals)) {
    data.samples = 0;
    data.renderTime = 0.0f;
    data.reset = true;
  }
  ImGui::Checkbox("Apply Gamma Correction", &data.gammaCorrect);
}


void handleSceneChanges(Scene* scene, PTData& data) {
  ImGui::Text("Load a new scene...");
  void(Scene::*populateFunc)(void) = NULL;

  if (ImGui::Button("Random Spheres")) {
    populateFunc = &Scene::populateRandomScene;
  }
  if (ImGui::Button("Random Spheres with Meshes")) {
    populateFunc = &Scene::populateComplexMesh;
  }
  if (ImGui::Button("Cornell Box")) {
    populateFunc = &Scene::populateCornellBox;
  }
  if (ImGui::Button("Cornell Box with Mirrors")) {
    populateFunc = &Scene::populateCornellBoxMetal;
  }
  if (ImGui::Button("Cornell Box with Glass")) {
    populateFunc = &Scene::populateCornellBoxDialectric;
  }
  if (ImGui::Button("BVH Test")) {
    populateFunc = &Scene::populateBVHTest;
  }

  if (populateFunc != NULL) {
    scene->freeFromGPU(data);
    scene->freeAll();

    (scene->*populateFunc)();
    scene->copyToGPU(data);

    data.samples = 0;
    data.renderTime = 0.0f;
    data.reset = true;
  }
}