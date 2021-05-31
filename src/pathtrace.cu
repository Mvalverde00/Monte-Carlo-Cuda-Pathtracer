#include "pathtrace.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "helper_cuda.h"
#include "camera.cuh"

surface<void, cudaSurfaceType2D> surf;



__global__ void writeColors(unsigned int width, unsigned int height, PTData args) {
  unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) {
    Ray r = args.cam->getRay(x, y, width, height, &args.rand[y * width + x]);

    HitRecord hr;
    hr.isHit = false;
    args.sph->hit(r, 1.0, 99999.0, hr);

    glm::vec3 dir = (glm::normalize(r.dir) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;
    if (hr.isHit) {
      dir = (glm::normalize(hr.normal) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;
    }

    if (args.reset) {
      args.accum[y * width + x] = dir;
    } else {
      args.accum[y * width + x] += dir;
    }
    glm::vec3 res = args.accum[y * width + x] / float(args.samples);
    uchar4 color = make_uchar4(255 * res.x, 255 * res.y, 255 * res.z, 255);
    surf2Dwrite(color, surf, x * sizeof(color), y, cudaBoundaryModeZero);
  }
}

void drawToScreen(int XRES, int YRES, cudaArray_const_t array, PTData& args) {
  CUDA_CALL(cudaBindSurfaceToArray(surf, array));
  const int blockX = 16;
  const int blockY = 16;
  dim3 blockSize(blockX, blockY);
  dim3 gridSize((XRES+ blockX - 1) / blockX, (YRES+ blockY - 1) / blockY);
  writeColors<<<gridSize, blockSize>>>((unsigned int)XRES, (unsigned int)YRES, args);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
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