#ifndef __PATHTRACE_H__
#define __PATHTRACE_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "camera.cuh"
#include "hittable.cuh"


struct PTData {
  curandState *rand;
  glm::vec3 *accum;

  // Scene Parameters
  int samples;
  bool reset; // reset the accumulator
  Sphere* sph;
  // int n_sphs;
  // Material* mats;
  // int n_mats;
  Camera *cam;
};

void initRandom(int XRES, int YRES, curandState* d_curand_state);

void drawToScreen(int XRES, int YRES, cudaArray_const_t array, PTData& args);

#endif