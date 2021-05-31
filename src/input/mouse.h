#ifndef __MOUSE_H__
#define __MOUSE_H__

#include <SDL.h>
#include <array>

class Mouse {
  // SDL defines mouse button as a Uint8, so 256 values can cover all possible buttons.
  std::array<bool, 256> buttonPressed;

  int dx;
  int dy;

public:

  Mouse();

  void processEvent(SDL_Event& event);
  bool isPressed(Uint8 button);

  int getDx();
  int getDy();

  /* Reset the position of the mouse, so that new mouse motions are calculated from (0,0) */
  void reset();
};
#endif
