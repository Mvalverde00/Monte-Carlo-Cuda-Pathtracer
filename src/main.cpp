#include "window.h"

#include "opengl_all.h"
#include "pathtrace.cuh"
#include "helper_cuda.h"
#include "opengl_interface.h"
#include "camera.cuh"
#include "hittable.cuh"
#include "input/mouse.h"
#include "input/keyboard.h"
#include "input/controls.h"
#include "scene.h"


#include <iostream>
#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl3.h>
#include <curand_kernel.h>


int main(int argc, char* argv) {
  size_t XRES = 1280;
  size_t YRES = 720;

  cudaGLSetGLDevice(0); // Assuming there is only 1 GPU, which there is on my system
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER);
  Window *window = new Window("SDL2 + CUDA", XRES, YRES);

  OpenGLInterface *openGL = new OpenGLInterface(XRES, YRES);

  // TODO: Setup camera, scene
  glm::mat4 rot = Camera::lookAt(glm::vec3(13, 2, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  Frustum frust(double(XRES) / YRES, 50.0, 2.0);
  Camera* cam = new Camera(glm::vec3(13, 2, 3), rot, frust);

  Camera* d_cam;
  CUDA_CALL(cudaMalloc(&d_cam, sizeof(Camera)));
  CUDA_CALL(cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice));

  curandState* d_rand_state;
  CUDA_CALL(cudaMalloc(&d_rand_state, XRES * YRES * sizeof(curandState)));
  initRandom(XRES, YRES, d_rand_state);

  glm::vec3* d_accum;
  CUDA_CALL(cudaMalloc(&d_accum, XRES * YRES * sizeof(glm::vec3)));
  CUDA_CALL(cudaMemset(d_accum, 0.0f, XRES * YRES * 3));

  PTData pathData(d_rand_state, d_accum, d_cam);

  Scene* scene = new Scene("F:\\cs179\\CudaPathTracer\\resources\\");
  /*
  int diffuseGray = scene->addMat(makeLambertian(glm::vec3(0.7, 0.7, 0.7)));
  int metallicCaltech = scene->addMat(makeMetal(glm::vec3(1.0, 0.4235, 0.0471)));
  int glass = scene->addMat(makeDialectric(1.3f));
  scene->addSphere(Sphere(glm::vec3(0.0, 0.0, -2.0), 1.0f, diffuseGray));
  scene->addSphere(Sphere(glm::vec3(2.0, 0.0, -2.0), 1.0f, metallicCaltech));
  scene->addSphere(Sphere(glm::vec3(-2.0, 0.0, -2.0), 1.0f, glass));
  scene->addSphere(Sphere(glm::vec3(0.0, -1001.0, -2.0), 1000.0f, diffuseGray));
  scene->copyToGPU(pathData);*/
  scene->populateComplexMesh();
  scene->copyToGPU(pathData);

  Mouse mouse = Mouse();
  Keyboard keyboard = Keyboard();

  SDL_Event event;
  bool running = true;
  Uint64 now;
  Uint64 prev = SDL_GetPerformanceCounter();
  Uint64 performanceFreq = SDL_GetPerformanceFrequency();
  while (running) {

    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);

      mouse.processEvent(event);
      keyboard.processEvent(event);

      if (event.type == SDL_QUIT) {
        running = false;
      }
    }

    // Begin new ImGUI frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window->_sdl_window);
    ImGui::NewFrame();


    now = SDL_GetPerformanceCounter();
    float dt = double(now - prev) / performanceFreq;
    prev = now;
    // handle movement, camera panning
    updateCamera(keyboard, mouse, cam, pathData, dt);

    displayCameraStats(cam, pathData);

    // Process rays and draw to screen with cuda
    openGL->beginCuda();
    pathData.samples += 1;
    pathData.renderTime += dt;
    drawToScreen(XRES, YRES, openGL->array_ptr, pathData);
    openGL->endCuda();

    //std::cout << "Frametime: " << (1.0 / ImGui::GetIO().Framerate) << "\n";

    openGL->blit(XRES, YRES);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
      std::cout << "OpenGL error!" << gluErrorString(err) << "\n";
    }

    // Render ImGUI widgets
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(window->_sdl_window);

    pathData.reset = false;
    mouse.reset();
  }

  CUDA_CALL(cudaFree(d_cam));
  CUDA_CALL(cudaFree(d_accum));
  CUDA_CALL(cudaFree(d_rand_state));
  scene->freeAll();
  scene->freeFromGPU(pathData);
  delete scene;
  delete cam;
  delete openGL;
  delete window;
}