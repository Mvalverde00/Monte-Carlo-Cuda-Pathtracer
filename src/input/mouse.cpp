#include "mouse.h"


Mouse::Mouse() {
  // Value-initialize array to false.
  buttonPressed = std::array<bool, 256>();
  dx = dy = 0;
};

void Mouse::processEvent(SDL_Event& event) {
  switch (event.type) {
  case SDL_MOUSEMOTION:
    dx += event.motion.xrel;
    dy += event.motion.yrel;
    break;
  case SDL_MOUSEBUTTONDOWN:
  case SDL_MOUSEBUTTONUP:
    buttonPressed[event.button.button] = event.button.state == SDL_PRESSED;
    break;
  default:
    break;
  }
}

bool Mouse::isPressed(Uint8 button) {
  return buttonPressed[button];
}

int Mouse::getDx() {
  return dx;
}

int Mouse::getDy() {
  return dy;
}

void Mouse::reset() {
  dx = dy = 0;
}
