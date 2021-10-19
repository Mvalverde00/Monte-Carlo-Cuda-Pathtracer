#include "pathtrace.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "helper_cuda.h"
#include "camera.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "mesh.cuh"
#include "bvh.cuh"

surface<void, cudaSurfaceType2D> surf;


PTData::PTData(curandState* d_rand, glm::vec3* d_accum, Camera* d_cam) {
  rand = d_rand;
  accum = d_accum;
  cam = d_cam;

  samples = 0;
  renderTime = 0.0f;
  nrays = NULL;
  reset = false;
  showNormals = false;
  gammaCorrect = false;

  background = glm::vec3(1, 1, 1);
  n_sphs = n_meshes = 0;
  sph = NULL;
  tris = NULL;
  meshes = NULL;
  mats = NULL;
  nodes = NULL;
}



__device__ bool intersectSpheres(const Ray& r, float t_min, float t_max, HitRecord& hr, Sphere* sphs, int n_sphs) {
  HitRecord temp;
  bool hit = false;
  hr.t = t_max;

  for (int i = 0; i < n_sphs; i++) {
    if (sphs[i].hit(r, t_min, t_max, temp) && temp.t < hr.t) {
      hr = temp;
      hit = true;
    }
  }

  return hit;
}


__device__ bool intersectMeshes(const Ray& r, float t_min, float t_max, HitRecord& hr, PTData args) {
  HitRecord temp;
  bool hit = false;
  hr.t = temp.t = t_max;

  for (int i = 0; i < (args).n_meshes; i++) {
    if (args.meshes[i].hit(r, t_min, t_max, temp, args.nodes, args.tris) && temp.t < hr.t) {
      hr = temp;
      hr.matIdx = args.meshes[i].matIdx;
      hit = true;
    }
  }

  return hit;
}

__device__ bool intersectScene(const Ray& r, float t_min, float t_max, HitRecord& hr, PTData args) {
  HitRecord sph, mesh;
  bool hit = intersectSpheres(r, t_min, t_max, sph, args.sph, args.n_sphs);
  bool hit2 = intersectMeshes(r, t_min, t_max, mesh, args);

  if (sph.t < mesh.t)
    hr = sph;
  else
    hr = mesh;

  return hit || hit2;
}

/* Computes the color received by a given ray by exploring a random path
 * through the scene */
__device__ glm::vec3 rayColor(Ray& r, HitRecord& hr, PTData args, curandState* rand) {
  glm::vec3 accumulate = glm::vec3(1, 1, 1);
  // Warp divergence as some rays terminate before others.
  for (int i = 0; i < 16; i++) {

    // Log that we are tracing another ray.
    args.nrays[(threadIdx.x + blockDim.x * blockIdx.x) + (threadIdx.y + blockDim.y * blockIdx.y) * 1280] += 1;

    if (intersectScene(r, 0.001f, 999999.0f, hr, args)) {
      if (args.showNormals)
        return (glm::normalize(hr.normal) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;



      glm::vec3 emitted = args.mats[hr.matIdx].emit();
      glm::vec3 attenuation;
      // Strong possibility for warp divergence here.  Although only currently 4 material types,
      // so max divergence of 4 in worst case.
      if (args.mats[hr.matIdx].scatter(r, hr, attenuation, rand)) {
        accumulate *= attenuation;
      } else {
        return accumulate * emitted;
      }

    } else {
      // Ray shoots off into background
      return accumulate * args.background;
    }
  }

  // Ray got stuck, never escaped.
  return glm::vec3(0, 0, 0);
}

/* Same as rayColor, but instead of returning immediately upon finishing a path,
 * we start tracing another path, keeping warp utilization at 100%.  However,
 * this means every warp now executes the full 16 iterations, slowing per-frame
 * performance a bit, but overall increasing the number of rays cast per second
 * and thus decreasing render time. */
__device__ glm::vec3 rayColorLessDivergence(Ray& r, HitRecord& hr, PTData args, curandState* rand) {
  glm::vec3 finalColor = glm::vec3(0, 0, 0);
  glm::vec3 accumulate = glm::vec3(1, 1, 1);
  int samples = 0;
  const Ray original = r;
  int useful_rays_traced = 0;

  // Less impact from warp divergence since every thread does a full 
  // 16 iterations.  Some work will be wasted, but on the whole
  // performance increases.
  for (int i = 0; i < 16; i++) {

    if (intersectScene(r, 0.001f, 999999.0f, hr, args)) {
      if (args.showNormals)
        return (glm::normalize(hr.normal) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;

      glm::vec3 emitted = args.mats[hr.matIdx].emit();
      glm::vec3 attenuation;
      // Strong possibility for warp divergence here.  Although only currently 4 material types,
      // so max divergence of 4 in worst case.
      if (args.mats[hr.matIdx].scatter(r, hr, attenuation, rand)) {
        accumulate *= attenuation;
      }
      else {
        // Ray was absorbed.  Path terminated
        finalColor += accumulate * emitted;
        accumulate = glm::vec3(1, 1, 1);
        samples += 1;
        r = original;
        useful_rays_traced = i + 1;
      }

    }
    else {
      // Ray shoots off into background. Path terminated
      finalColor += accumulate * args.background;
      accumulate = glm::vec3(1, 1, 1);
      samples += 1;
      r = original;
      useful_rays_traced = i + 1;
    }
  }

  // Log number of rays traced for performance testing
  args.nrays[(threadIdx.x + blockDim.x * blockIdx.x) + (threadIdx.y + blockDim.y * blockIdx.y) * 1280] += useful_rays_traced;

  return finalColor / float(max(samples, 1));
}


__global__ void cudaTakeSampleAndDraw(unsigned int width, unsigned int height, PTData args) {
  unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) {
    Ray r = args.cam->getRay(x, y, width, height, &args.rand[y * width + x]);
    r = Ray(r.at(1.0f), r.dir);
    HitRecord hr;

    glm::vec3 sample_value = rayColor(r, hr, args, &args.rand[y * width + x]);

    // All threads follow same branch so no warp divergence
    
    glm::vec3 res;
    if (args.reset) {
      args.r[y * width + x] = sample_value.r;
      args.g[y * width + x] = sample_value.g;
      args.b[y * width + x] = sample_value.b;
      res = glm::vec3(sample_value.r, sample_value.g, sample_value.b);
    } else {
      // Each of args.r/b/g can be written/read to by a single coalesced global read/write
      res.r = args.r[y * width + x] + sample_value.r;
      res.g = args.g[y * width + x] + sample_value.g;
      res.b = args.b[y * width + x] + sample_value.b;
      args.r[y * width + x] = res.r;
      args.g[y * width + x] = res.g;
      args.b[y * width + x] = res.b;
      res = res / float(args.samples);
    }

    if (args.gammaCorrect)
      res = glm::clamp(glm::sqrt(res), 0.0f, 1.0f);
    else
      res = glm::clamp(res, 0.0f, 1.0f);
    
    /*
    if (args.reset) {
      args.accum[y * width + x] = sample_value;
    } else {
      args.accum[y * width + x] += sample_value;
    }
    glm::vec3 res = glm::clamp(args.accum[y * width + x] / float(args.samples), 0.0f, 1.0f);*/

    uchar4 color = make_uchar4(255 * res.x, 255 * res.y, 255 * res.z, 255);
    surf2Dwrite(color, surf, x * sizeof(color), y, cudaBoundaryModeZero);
  }
}

void takeSampleAndDraw(int XRES, int YRES, cudaArray_const_t array, PTData& args) {
  CUDA_CALL(cudaBindSurfaceToArray(surf, array));
  const int blockX = 32;
  const int blockY = 3;
  dim3 blockSize(blockX, blockY);
  dim3 gridSize((XRES+ blockX - 1) / blockX, (YRES+ blockY - 1) / blockY);
  cudaTakeSampleAndDraw<<<gridSize, blockSize>>>((unsigned int)XRES, (unsigned int)YRES, args);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  

  //int bs, gs;
  //cudaOccupancyMaxPotentialBlockSize(&bs, &gs, writeColors);
  //std::cout << bs << ", " << gs << "\n";
}







__global__ void cudaInitRandom(int width, int height, curandState* d_rand) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    curand_init(2021, idx, 0, &d_rand[idx]);
  }
}


void initRandom(int XRES, int YRES, curandState* d_curand_state) {
  const int blockX = 32;
  const int blockY = 32;
  dim3 blockSize(blockX, blockY);
  dim3 gridSize((XRES + blockX - 1) / blockX, (YRES + blockY - 1) / blockY);
  cudaInitRandom<<<gridSize, blockSize>>>(XRES, YRES, d_curand_state);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}