#include "window.h"

#include <cassert>
#include <iostream>

#include "opengl_all.h" // must be included BEFORE imgui

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_opengl3.h>

Window::Window(std::string name, int width, int height) {
  // Set OpenGL 4.2 as target version.
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

  _sdl_window = SDL_CreateWindow(name.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);
  _width = width;
  _height = height;

  _context = SDL_GL_CreateContext(_sdl_window);

  //SDL_SetRelativeMouseMode(SDL_TRUE); // Captures mouse
  SDL_GL_SetSwapInterval(1); // Enable VSync

  glewInit();
  glViewport(0, 0, width, height);

  glEnable(GL_DEPTH_TEST); // Use depth test

  // For now, we will setup ImGUI by default.  In the future, make its use toggleable
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplSDL2_InitForOpenGL(_sdl_window, _context);
  ImGui_ImplOpenGL3_Init("#version 420"); // Eventually this should not be hardcoded
}

Window::Window() {};

Window::~Window() {
  std::cout << "Destroying window!\n";
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(_context);
  SDL_DestroyWindow(_sdl_window);
}