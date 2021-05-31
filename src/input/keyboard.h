#ifndef __KEYBOARD_H__
#define __KEYBOARD_H__

#include <SDL.h>
#include <unordered_map>


class Keyboard {
  std::unordered_map<SDL_Keycode, bool> currState;
  std::unordered_map<SDL_Keycode, bool> prevState;

public:
  Keyboard();

  void processEvent(SDL_Event& event);

  void beginTick();

  /* Whether the key is being pressed right now */
  bool isKeyDown(SDL_Keycode keycode);

  /* Whether the key was just pressed or just released */
  bool keyPressed(SDL_Keycode keycode);
  bool keyReleased(SDL_Keycode keycode);


};

#endif
