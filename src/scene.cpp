#include "scene.h"

#include <iostream>
#include <glm/gtx/transform.hpp>

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
  //srand((unsigned int)time(0));
  srand((unsigned int)4);

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
  glm::mat4 xform = glm::scale(glm::vec3(2, 2, 2)) * glm::rotate(glm::radians(90.0f), glm::vec3(0, 1, 0));
  addMesh("bunny.obj", mat1, xform);


  int mat2 = addMat(makeLambertian(glm::vec3(0.4, 0.2, 0.1)));
  glm::mat4 umbreonXform = glm::translate(glm::vec3(-4, 1.0, 0)) * glm::scale(glm::vec3(1.2, 1.2, 1.2));
  addMesh("umbreon.obj", mat2, umbreonXform);

  int mat3 = addMat(makeMetal(glm::vec3(0.7, 0.6, 0.5), 0.0));
  glm::mat4 bunnyXform = glm::translate(glm::vec3(4, 0, 0)) * glm::scale(glm::vec3(2, 2, 2)) * glm::rotate(glm::radians(90.0f), glm::vec3(0, 1, 0));
  addMesh("bunny.obj", mat3, bunnyXform);
  //glm::mat4 bunnyXform = glm::translate(glm::vec3(4, 0, 0)) * glm::scale(glm::vec3(2.0 /17.0f, 2.0 / 17.0f, 2.0 / 17.0f)) * glm::rotate(glm::radians(90.0f), glm::vec3(-1, 0, 0));
  //addMesh("teapot.obj", mat3, bunnyXform);
  //addSphere(Sphere(glm::vec3(4, 1, 0), 1, mat3));

}


void Scene::populateCornellBox() {
  int redDiffuse = addMat(makeLambertian(glm::vec3(1.0, 0.6, 0.6)));
  int grayDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 0.6)));
  int blueDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 1.0)));
  int whiteLight = addMat(makeLight(glm::vec3(15.0, 15.0, 15.0)));

  float scale = 100.0;
  glm::mat4 wallScale = glm::scale(glm::vec3(scale, scale, 1));
  glm::mat4 floorTMat = glm::translate(glm::vec3(-scale / 2, 0, -scale / 2)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * wallScale;
  addMesh("square.obj", grayDiffuse, floorTMat);
  glm::mat4 ceilingTMat = glm::translate(glm::vec3(0, 45, 0)) * floorTMat;
  addMesh("square.obj", grayDiffuse, ceilingTMat);

  glm::mat4 backTMat = glm::translate(glm::vec3(-scale / 2, 0, 15)) * wallScale;
  addMesh("square.obj", grayDiffuse, backTMat);

  glm::mat4 leftTMat = glm::translate(glm::vec3(-25, 0, 15)) * glm::rotate(glm::radians(90.0f), glm::vec3(0, 1, 0)) * wallScale;
  addMesh("square.obj", redDiffuse, leftTMat);
  glm::mat4 rightTMat = glm::translate(glm::vec3(50, 0, 0)) * leftTMat;
  addMesh("square.obj", blueDiffuse, rightTMat);

  glm::mat4 tallCubeTMat = glm::translate(glm::vec3(-10, 10, 0)) * glm::rotate(glm::radians(45.0f), glm::vec3(0, 1, 0)) * glm::scale(glm::vec3(7, 15, 7));
  addMesh("cube.obj", grayDiffuse, tallCubeTMat);

  glm::mat4 shortCubeTMat = glm::translate(glm::vec3(10, 6, -10)) * glm::rotate(glm::radians(45.0f), glm::vec3(0, 1, 0)) * glm::scale(glm::vec3(6, 8, 6));
  addMesh("cube.obj", grayDiffuse, shortCubeTMat);

  glm::mat4 lightTMat = floorTMat = glm::translate(glm::vec3(-15.0 / 2.0, 44.9, -15.0 / 2.0)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::vec3(15, 15, 1));
  addMesh("square.obj", whiteLight, lightTMat);
}