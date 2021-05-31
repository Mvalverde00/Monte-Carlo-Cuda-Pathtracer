#include "opengl_interface.h"

#include "helper_cuda.h"

OpenGLInterface::OpenGLInterface(size_t x, size_t y) : renderXRes(x), renderYRes(y) {
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);

  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, renderXRes, renderYRes);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);


  CUDA_CALL(cudaGraphicsGLRegisterImage(&resource, rbo, GL_RENDERBUFFER,
                    cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard));
}

OpenGLInterface::~OpenGLInterface() {
  CUDA_CALL(cudaGraphicsUnregisterResource(resource));

  glDeleteFramebuffers(1, &fbo);
  glDeleteRenderbuffers(1, &rbo);
}


void OpenGLInterface::beginCuda() {
  CUDA_CALL(cudaGraphicsMapResources(1, &resource));
  CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&array_ptr, resource, 0, 0));
  CUDA_CALL(cudaDeviceSynchronize());
}
void OpenGLInterface::endCuda() {
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaGraphicsUnmapResources(1, &resource));
}

const GLuint SCREEN_FBO = 0;
void OpenGLInterface::blit(size_t& XRES, size_t& YRES) {
  glBlitNamedFramebuffer(fbo, SCREEN_FBO,
    0, 0, renderXRes, renderYRes,
    0, 0, XRES, YRES,
    GL_COLOR_BUFFER_BIT, GL_NEAREST);
}