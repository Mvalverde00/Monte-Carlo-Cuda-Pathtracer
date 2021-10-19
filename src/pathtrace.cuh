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
class BVHNode;


/* Stores all of the data to be passed to cuda kernel
 * which is necessary to render the scene */
struct PTData {
  curandState *rand;
  glm::vec3 *accum;
  Camera *cam;
  float *r;
  float *g;
  float *b;

  // Scene Info
  int samples;
  float renderTime;
  int *nrays;
  bool reset; // reset the accumulator
  bool showNormals; // Display normals of objects.
  bool gammaCorrect; // Use gamma correction or not.

  // Scene Description
  glm::vec3 background;
  Sphere *sph;
  int n_sphs;
  Triangle *tris;
  MeshInstance *meshes;
  int n_meshes;
  Material *mats;
  BVHNode *nodes;

  PTData(curandState* d_rand, glm::vec3* d_accum, Camera* d_cam);
};

/* Initialize curandState for each thread*/
void initRandom(int XRES, int YRES, curandState* d_curand_state);


/* Progressively renders one more sample per pixel for each pixel on the screen,
 * in a private accumulation buffer, and then copies the result to the provided
 * cudaArray_const_t */
void takeSampleAndDraw(int XRES, int YRES, cudaArray_const_t array, PTData& args);

#endif