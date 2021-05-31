#ifndef __WINDOW_H__
#define __WINDOW_H__

#include <string>
#include <SDL.h>
#undef main

class Window {
  public:
    Window(std::string name, int width, int height);
    Window();
    ~Window();

    SDL_Window* _sdl_window;
    SDL_GLContext _context;
    int _width, _height;
};

#endif