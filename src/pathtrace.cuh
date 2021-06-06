#ifndef __PATHTRACE_H__
#define __PATHTRACE_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>


// Forward Declarations
class Sphere;
class Material;
class Camera;
class Triangle;
class MeshInstance;


struct PTData {
  curandState *rand;
  glm::vec3 *accum;
  Camera *cam;

  // Scene Info
  int samples;
  float renderTime;
  bool reset; // reset the accumulator
  bool showNormals; // Display normals of objects.

  // Scene Description
  Sphere *sph;
  int n_sphs;
  Triangle *tris;
  MeshInstance *meshes;
  int n_meshes;
  Material *mats;

  PTData(curandState* d_rand, glm::vec3* d_accum, Camera* d_cam);
};

void initRandom(int XRES, int YRES, curandState* d_curand_state);

void drawToScreen(int XRES, int YRES, cudaArray_const_t array, PTData& args);

#endif