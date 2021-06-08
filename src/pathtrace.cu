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

surface<void, cudaSurfaceType2D> surf;


PTData::PTData(curandState* d_rand, glm::vec3* d_accum, Camera* d_cam) {
  rand = d_rand;
  accum = d_accum;
  cam = d_cam;

  samples = 0;
  renderTime = 0.0f;
  reset = false;
  showNormals = false;

  n_sphs = n_meshes = 0;
  sph = NULL;
  tris = NULL;
  meshes = NULL;
  mats = NULL;
}



__device__ void intersectSpheres(const Ray& r, float t_min, float t_max, HitRecord& hr, Sphere* sphs, int n_sphs) {
  HitRecord temp;
  temp.isHit = hr.isHit = false;
  hr.t = t_max;

  for (int i = 0; i < n_sphs; i++) {
    sphs[i].hit(r, t_min, t_max, temp);

    if (temp.isHit && temp.t < hr.t) {
      hr = temp;
    }
  }
}


__device__ void intersectMeshes(const Ray& r, float t_min, float t_max, HitRecord& hr, PTData args) {
  HitRecord temp;
  temp.isHit = hr.isHit = false;
  hr.t = t_max;

  for (int i = 0; i < (args).n_meshes; i++) {
    args.meshes[i].hit(r, t_min, t_max, temp, args.tris);
    temp.matIdx = args.meshes[i].matIdx;
    if (temp.isHit && temp.t < hr.t) {
      hr = temp;
    }
  }
}

__device__ void intersectScene(const Ray& r, float t_min, float t_max, HitRecord& hr, PTData args) {
  HitRecord sph, mesh;
  intersectSpheres(r, t_min, t_max, sph, args.sph, args.n_sphs);
  intersectMeshes(r, t_min, t_max, mesh, args);

  if (sph.t < mesh.t)
    hr = sph;
  else
    hr = mesh;
}

__device__ glm::vec3 rayColor(Ray& r, HitRecord& hr, PTData& args, curandState* rand) {
  glm::vec3 accumulate = glm::vec3(1, 1, 1);
  for (int i = 0; i < 10; i++) {
    //intersectSpheres(r, 0.001f, 999999.0f, hr, args.sph, args.n_sphs);
    intersectScene(r, 0.001f, 999999.0f, hr, args);

    if (hr.isHit) {
      if (args.showNormals)
        return (glm::normalize(hr.normal) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;

      glm::vec3 emitted = args.mats[hr.matIdx].emit();
      glm::vec3 attenuation;
      if (args.mats[hr.matIdx].scatter(r, hr, attenuation, rand)) {
        accumulate *= attenuation;
      } else {
        return accumulate * emitted;
      }

    } else {
      // Ray shoots off into background
      return accumulate * glm::vec3(1, 1, 1);
    }
  }

  // Ray got stuck, never escaped.
  return glm::vec3(0, 0, 0);
}

__global__ void writeColors(unsigned int width, unsigned int height, PTData args) {
  unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) {
    Ray r = args.cam->getRay(x, y, width, height, &args.rand[y * width + x]);
    r = Ray(r.at(1.0f), r.dir);
    HitRecord hr;

    glm::vec3 sample_value = rayColor(r, hr, args, &args.rand[y * width + x]);

    /*
    if (args.reset) {
      args.r[y * width + x] = sample_value.r;
      args.g[y * width + x] = sample_value.g;
      args.b[y * width + x] = sample_value.b;
    } else {
      args.r[y * width + x] += sample_value.r;
      args.g[y * width + x] += sample_value.g;
      args.b[y * width + x] += sample_value.b;
    }
    glm::vec3 res = glm::clamp(glm::vec3(args.r[y * width + x], args.g[y * width + x], args.b[y * width + x]) / float(args.samples), 0.0f, 1.0f);
    */
    
    if (args.reset) {
      args.accum[y * width + x] = sample_value;
    } else {
      args.accum[y * width + x] += sample_value;
    }
    glm::vec3 res = glm::clamp(args.accum[y * width + x] / float(args.samples), 0.0f, 1.0f);
    uchar4 color = make_uchar4(255 * res.x, 255 * res.y, 255 * res.z, 255);
    surf2Dwrite(color, surf, x * sizeof(color), y, cudaBoundaryModeZero);
  }
}

void drawToScreen(int XRES, int YRES, cudaArray_const_t array, PTData& args) {
  CUDA_CALL(cudaBindSurfaceToArray(surf, array));
  const int blockX = 32;
  const int blockY = 3;
  dim3 blockSize(blockX, blockY);
  dim3 gridSize((XRES+ blockX - 1) / blockX, (YRES+ blockY - 1) / blockY);
  writeColors<<<gridSize, blockSize>>>((unsigned int)XRES, (unsigned int)YRES, args);
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