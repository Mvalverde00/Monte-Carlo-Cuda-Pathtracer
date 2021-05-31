#ifndef __CONTROLS_H__
#define __CONTROLS_H__

#include "input/keyboard.h"
#include "input/mouse.h"
#include "pathtrace.cuh"
#include "camera.cuh"

/* Functions in this file are responsible for displaying and
 * managing the camera state */


/* Updates the camera position given the keyboard presses and mouse movements for
 * the last time interval dt.  Also updates PTData to reflect when the view has been changed
 * and the number of samples needs to be reset. */
void updateCamera(Keyboard& keyboard, Mouse& mouse, Camera* cam, PTData& data, float dt);

/* Displays the current camera position and direction, as well as the number of samples
 * accumulated */
void displayCameraStats(Camera* cam, PTData& data);

#endif