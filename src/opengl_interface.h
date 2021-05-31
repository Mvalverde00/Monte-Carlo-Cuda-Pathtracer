#ifndef __OPENGL_INTERFACE_H__
#define __OPENGL_INTERFACE_H__

#include "opengl_all.h"
#include <cuda_gl_interop.h>

class OpenGLInterface {
  // Frame and Render buffers
  GLuint fbo, rbo;
  cudaGraphicsResource* resource;
  // The internal rendering x/y resolution to be used. Can be
  // larger or smaller than the physical window resolution,
  // Will eventually be useful if I support exporting images,
  // so you can e.g. render a 4k picture and export it while
  // working on a 1080p monitor.
  size_t renderXRes, renderYRes;

public:
  // Pointer to the rbo data, which can be overwritten.
  // Do not use unless beginCuda() has been called.
  cudaArray_t array_ptr;

  // The render x/y resolutions
  OpenGLInterface(size_t x, size_t y);
  ~OpenGLInterface(); // need to free buffers

  // Populates array_ptr with the necessary data to write to the
  // rbo.  Once called, OpenGL should not attempt to use the
  // rbo until endCuda() is called
  void beginCuda();

  // Unclaims array_ptr so that OpenGL can access the data for rendering
  // to screen. 
  void endCuda();

  // Draws the fbo to the screen.
  void blit(size_t& XRES, size_t& YRES);
};


#endif