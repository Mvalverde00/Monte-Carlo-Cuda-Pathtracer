#include "scene.h"

#include <iostream>

#include "helper_cuda.h"
#include "pathtrace.cuh"

Mesh Scene::loadMesh(std::string fpath) {
  try {
    return Mesh(fpath, tris);
  }
  catch (std::exception& e) {
    std::cerr << "Aborting due to failure to load resources\n";
    exit(1);
  }
}

void Scene::addSphere(Sphere& sph) {
  spheres.push_back(sph);
}

int Scene::addMat(Material& mat) {
  materials.push_back(mat);

  return int(materials.size()) - 1;
}

void Scene::addMesh(std::string fpath, int matIdx, glm::mat4 tMat) {
  // We already have a copy of the base mesh
  if (meshes.find(fpath) != meshes.end()) {
    instances.push_back(MeshInstance(meshes[fpath], matIdx, tMat));
  }
  else {
    Mesh base = loadMesh(resourceDir + fpath);
    meshes.insert({ fpath, base });
    instances.push_back(MeshInstance(base, matIdx, tMat));
  }

}

void Scene::freeAll() {
  spheres.clear();
  materials.clear();
  instances.clear();
  meshes.clear();
}


void Scene::copyToGPU(PTData& data) {
  /* TODO: I would like to be able to error check these with cuda call, but doing so
   * throws an error for some reason */

  cudaMalloc(&data.sph, sizeof(Sphere) * spheres.size());
  cudaMemcpy(data.sph, &spheres[0], sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice);
  data.n_sphs = int(spheres.size());

  cudaMalloc(&data.mats, sizeof(Material) * materials.size());
  cudaMemcpy(data.mats, &materials[0], sizeof(Material) * materials.size(), cudaMemcpyHostToDevice);

  
  cudaMalloc(&data.tris, sizeof(Triangle) * tris.size());
  cudaMemcpy(data.tris, &tris[0], sizeof(Triangle) * tris.size(), cudaMemcpyHostToDevice);

  cudaMalloc(&data.meshes, sizeof(MeshInstance) * instances.size());
  cudaMemcpy(data.meshes, &instances[0], sizeof(MeshInstance) * instances.size(), cudaMemcpyHostToDevice);
  data.n_meshes = int(instances.size());
  
}

void Scene::freeFromGPU(PTData& data) {
  cudaFree(data.sph);
  cudaFree(data.mats);
  
  cudaFree(data.tris);
  cudaFree(data.meshes);
}



float randFloat() {
  return float(rand()) / float(RAND_MAX);
}

float randFloat(float min, float max) {
  return min + (max - min) * randFloat();
}

void Scene::populateRandomScene() {
  srand((unsigned int)time(0));

  int groundMat = addMat(makeLambertian(glm::vec3(0.5, 0.5, 0.5)));
  addSphere(Sphere(glm::vec3(0.0, -1000.0, 0.0), 1000.0, groundMat));

  
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float chooseMat = randFloat();
      glm::vec3 center(a + 0.9 * randFloat(), 0.2, b + 0.9 * randFloat());

      if (glm::length(center - glm::vec3(4.0, 0.2, 0.0)) > 0.9) {
        int mat;
        if (chooseMat < 0.8) {
          glm::vec3 c = glm::vec3(randFloat(), randFloat(), randFloat());
          mat = addMat(makeLambertian(c));
        }
        else if (chooseMat < 0.95) {
          glm::vec3 c = glm::vec3(randFloat(0.5, 1.0), randFloat(0.5, 1.0), randFloat(0.5, 1.0));
          float fuzzing = randFloat(0.0, 0.5);
          mat = addMat(makeMetal(c, fuzzing));
        }
        else {
          mat = addMat(makeDialectric(1.5));
        }
        addSphere(Sphere(center, 0.2f, mat));
      }
    }
  }

  int mat1 = addMat(makeDialectric(1.5));
  addSphere(Sphere(glm::vec3(0, 1, 0), 1.0, mat1));

  int mat2 = addMat(makeLambertian(glm::vec3(0.4, 0.2, 0.1)));
  addSphere(Sphere(glm::vec3(-4, 1, 0), 1.0, mat2));

  int mat3 = addMat(makeMetal(glm::vec3(0.7, 0.6, 0.5), 0.0));
  addSphere(Sphere(glm::vec3(4, 1, 0), 1.0, mat3));
}

void Scene::populateComplexMesh() {
  srand((unsigned int)time(0));

  int groundMat = addMat(makeLambertian(glm::vec3(0.5, 0.5, 0.5)));
  addSphere(Sphere(glm::vec3(0.0, -1000.0, 0.0), 1000.0, groundMat));

  /*
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float chooseMat = randFloat();
      glm::vec3 center(a + 0.9 * randFloat(), 0.2, b + 0.9 * randFloat());

      if (glm::length(center - glm::vec3(4.0, 0.2, 0.0)) > 0.9) {
        int mat;
        if (chooseMat < 0.8) {
          glm::vec3 c = glm::vec3(randFloat(), randFloat(), randFloat());
          mat = addMat(makeLambertian(c));
        }
        else if (chooseMat < 0.95) {
          glm::vec3 c = glm::vec3(randFloat(0.5, 1.0), randFloat(0.5, 1.0), randFloat(0.5, 1.0));
          float fuzzing = randFloat(0.0, 0.5);
          mat = addMat(makeMetal(c, fuzzing));
        }
        else {
          mat = addMat(makeDialectric(1.5));
        }
        addSphere(Sphere(center, 0.2f, mat));
      }
    }
  }*/

  int mat1 = addMat(makeDialectric(1.5));
  //addSphere(Sphere(glm::vec3(0, 1, 0), 1.0, mat1));
  //addMesh("bunny.obj", mat1, glm::mat4(1.0f));

  int mat2 = addMat(makeLambertian(glm::vec3(0.4, 0.2, 0.1)));
  addSphere(Sphere(glm::vec3(-4, 1, 0), 1.0, mat2));

  int mat3 = addMat(makeMetal(glm::vec3(0.7, 0.6, 0.5), 0.0));
  addSphere(Sphere(glm::vec3(4, 1, 0), 1.0, mat3));

  addMesh("umbreon.obj", mat2, glm::mat4(1.0f));
}