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


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: CudaPathTracer.exe <RESOURCE_DIR>\n";
    exit(1);
  }

  size_t XRES = 1280;
  size_t YRES = 720;

  cudaGLSetGLDevice(0); // Assuming there is only 1 GPU, which there is on my system
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER);
  Window *window = new Window("SDL2 + CUDA Path Tracer.", XRES, YRES);

  OpenGLInterface *openGL = new OpenGLInterface(XRES, YRES);


  // Setup necessary device paramters
  curandState* d_rand_state;
  CUDA_CALL(cudaMalloc(&d_rand_state, XRES * YRES * sizeof(curandState)));
  initRandom(XRES, YRES, d_rand_state);

  glm::vec3* d_accum;
  CUDA_CALL(cudaMalloc(&d_accum, XRES * YRES * sizeof(glm::vec3)));
  CUDA_CALL(cudaMemset(d_accum, 0.0f, XRES * YRES * 3));

  float *d_r, *d_g, *d_b;
  CUDA_CALL(cudaMalloc(&d_r, XRES * YRES * sizeof(float)));
  CUDA_CALL(cudaMemset(d_r, 0.0f, XRES * YRES));
  CUDA_CALL(cudaMalloc(&d_g, XRES * YRES * sizeof(float)));
  CUDA_CALL(cudaMemset(d_g, 0.0f, XRES * YRES));
  CUDA_CALL(cudaMalloc(&d_b, XRES * YRES * sizeof(float)));
  CUDA_CALL(cudaMemset(d_b, 0.0f, XRES * YRES));

  int *d_nrays;
  CUDA_CALL(cudaMalloc(&d_nrays, XRES * YRES * sizeof(int)));
  CUDA_CALL(cudaMemset(d_nrays, 0, XRES * YRES * sizeof(int)));

  PTData pathData(d_rand_state, d_accum, NULL);
  pathData.r = d_r;
  pathData.g = d_g;
  pathData.b = d_b;
  pathData.nrays = d_nrays;

  Scene* scene = new Scene(std::string(argv[1]));
  scene->populateRandomScene();
  //scene->populateComplexMesh();
  //scene->populateCornellBox();
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
    updateCamera(keyboard, mouse, scene->cam, pathData, dt);
    displayCameraStats(scene->cam, pathData);
    handleSceneChanges(scene, pathData);

    // Process rays and draw to screen with cuda
    openGL->beginCuda();
    pathData.samples += 1;
    pathData.renderTime += dt;
    takeSampleAndDraw(XRES, YRES, openGL->array_ptr, pathData);
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
  

  // Print out ray statistics
  int* nrays = new int[XRES * YRES * sizeof(int)];
  CUDA_CALL(cudaMemcpy(nrays, d_nrays, XRES * YRES * sizeof(int), cudaMemcpyDeviceToHost));
  long long int total = 0;
  for (int i = 0; i < XRES * YRES; i++) {
    total += nrays[i];
  }
  std::cout << "Traced " << total << " rays in " << pathData.renderTime << " seconds\n";
  std::cout << " = " << (total / pathData.renderTime) / 1000000.f << " Mrays per second\n";
  delete nrays;
  
  CUDA_CALL(cudaFree(d_nrays));
  CUDA_CALL(cudaFree(d_r));
  CUDA_CALL(cudaFree(d_g));
  CUDA_CALL(cudaFree(d_b));
  CUDA_CALL(cudaFree(d_accum));
  CUDA_CALL(cudaFree(d_rand_state));
  scene->freeAll();
  scene->freeFromGPU(pathData);
  delete scene;
  delete openGL;
  delete window;
}