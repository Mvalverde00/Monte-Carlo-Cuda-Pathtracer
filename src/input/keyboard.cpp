#include "keyboard.h"


Keyboard::Keyboard() {}

void Keyboard::processEvent(SDL_Event& event) {
  switch (event.type) {
  case SDL_KEYDOWN:
    currState[event.key.keysym.sym] = true;
    break;
  case SDL_KEYUP:
    currState[event.key.keysym.sym] = false;
    break;
  default:
    break;
  }
}

void Keyboard::beginTick() {
  prevState = currState;
}


bool Keyboard::isKeyDown(SDL_Keycode keycode) {
  return currState[keycode];
}

bool Keyboard::keyPressed(SDL_Keycode keycode) {
  return currState[keycode] && !prevState[keycode];
}

bool Keyboard::keyReleased(SDL_Keycode keycode) {
  return prevState[keycode] && !currState[keycode];
}
