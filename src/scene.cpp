#include "scene.h"

#include <iostream>
#include <glm/gtx/transform.hpp>

#include "helper_cuda.h"
#include "pathtrace.cuh"
#include "camera.cuh"


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
    base.initAcceleration(nodes, tris);
    meshes.insert({ fpath, base });
    instances.push_back(MeshInstance(base, matIdx, tMat));
    std::cout << "offset " << base.bvhOffset << ". total nodes " << nodes.size() << " total tris " << tris.size() << "\n";
  }


}

void Scene::freeAll() {
  spheres.clear();
  materials.clear();
  instances.clear();
  meshes.clear();
  tris.clear();
  nodes.clear();

  // Push a blank since index 0 is meant to be reserved for null
  nodes.push_back(BVHNode());

  free(cam);
}


void Scene::copyToGPU(PTData& data) {
  /* TODO: I would like to be able to error check these with cuda call, but doing so
   * throws an error for some reason */
  data.background = background;

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
  
  cudaMalloc(&data.nodes, sizeof(BVHNode) * nodes.size());
  cudaMemcpy(data.nodes, &nodes[0], sizeof(BVHNode) * nodes.size(), cudaMemcpyHostToDevice);

  cudaMalloc(&data.cam, sizeof(Camera));
  cudaMemcpy(data.cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
}

void Scene::freeFromGPU(PTData& data) {
  cudaFree(data.sph);
  cudaFree(data.mats);
  
  cudaFree(data.tris);
  cudaFree(data.meshes);
  cudaFree(data.nodes);

  cudaFree(data.cam);
}



float randFloat() {
  return float(rand()) / float(RAND_MAX);
}

float randFloat(float min, float max) {
  return min + (max - min) * randFloat();
}

void Scene::populateRandomScene() {
  //srand((unsigned int)time(0));
  srand(0);

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

  glm::mat4 rot = Camera::lookAt(glm::vec3(13, 2, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  Frustum frust(aspectRatio, 50.0f, 2.0f);
  cam = new Camera(glm::vec3(13, 2, 3), rot, frust);
  background = glm::vec3(1, 1, 1);
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

  glm::mat4 rot = Camera::lookAt(glm::vec3(13, 2, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  Frustum frust(aspectRatio, 50.0f, 2.0f);
  cam = new Camera(glm::vec3(13, 2, 3), rot, frust);
  background = glm::vec3(1, 1, 1);
}


void Scene::populateCornellBox() {
  int redDiffuse = addMat(makeLambertian(glm::vec3(1.0, 0.6, 0.6)));
  int grayDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 0.6)));
  int blueDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 1.0)));
  int whiteLight = addMat(makeLight(glm::vec3(10.0, 10.0, 10.0)));

  int grayMetal = addMat(makeMetal(glm::vec3(0.6, 0.6, 0.6)));
  int redMetal = addMat(makeMetal(glm::vec3(1.0, 0.6, 0.6)));
  int blueMetal = addMat(makeMetal(glm::vec3(0.6, 0.6, 1.0)));
  
  addMesh("square.obj", grayDiffuse, glm::mat4(1.f));
  for (BVHNode& node : nodes) {
    std::cout << node.left << ", " << node.right << "\n";
    std::cout << node.bbox.minimum.x << ", " << node.bbox.minimum.y << ", " << node.bbox.minimum.z << "\n";
    std::cout << node.bbox.maximum.x << ", " << node.bbox.maximum.y << ", " << node.bbox.maximum.z << "\n";
  }

  for (Triangle& tri : tris) {
    std::cout << tri.v0.pos.x << ", " << tri.v0.pos.y << ", " << tri.v0.pos.z << "\n";
    std::cout << tri.v1.pos.x << ", " << tri.v1.pos.y << ", " << tri.v1.pos.z << "\n";
    std::cout << tri.v2.pos.x << ", " << tri.v2.pos.y << ", " << tri.v2.pos.z << "\n";
  }

  /*
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
  addMesh("square.obj", whiteLight, lightTMat);*/

  Frustum frust(aspectRatio, 75.0f, 1.0f);
  glm::vec3 pos(0, 20, -45);
  glm::mat4 rot = Camera::lookAt(pos, glm::vec3(0, 15, 0), glm::vec3(0, 1, 0));
  cam = new Camera(pos, rot, frust);
  background = glm::vec3(0, 0, 0);
}

void Scene::populateCornellBoxMetal() {
  int grayDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 0.6)));
  int whiteLight = addMat(makeLight(glm::vec3(10.0, 10.0, 10.0)));

  int grayMetal = addMat(makeMetal(glm::vec3(0.6, 0.6, 0.6)));
  int redMetal = addMat(makeMetal(glm::vec3(1.0, 0.6, 0.6)));
  int blueMetal = addMat(makeMetal(glm::vec3(0.6, 0.6, 1.0)));

  float scale = 100.0;
  glm::mat4 wallScale = glm::scale(glm::vec3(scale, scale, 1));
  glm::mat4 floorTMat = glm::translate(glm::vec3(-scale / 2, 0, -scale / 2)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * wallScale;
  addMesh("square.obj", grayDiffuse, floorTMat);
  glm::mat4 ceilingTMat = glm::translate(glm::vec3(0, 45, 0)) * floorTMat;
  addMesh("square.obj", grayDiffuse, ceilingTMat);

  glm::mat4 backTMat = glm::translate(glm::vec3(-scale / 2, 0, 15)) * wallScale;
  addMesh("square.obj", grayMetal, backTMat);

  glm::mat4 leftTMat = glm::translate(glm::vec3(-25, 0, 15)) * glm::rotate(glm::radians(90.0f), glm::vec3(0, 1, 0)) * wallScale;
  addMesh("square.obj", redMetal, leftTMat);
  glm::mat4 rightTMat = glm::translate(glm::vec3(50, 0, 0)) * leftTMat;
  addMesh("square.obj", blueMetal, rightTMat);

  glm::mat4 tallCubeTMat = glm::translate(glm::vec3(-10, 10, 0)) * glm::rotate(glm::radians(45.0f), glm::vec3(0, 1, 0)) * glm::scale(glm::vec3(7, 15, 7));
  addMesh("cube.obj", grayDiffuse, tallCubeTMat);

  glm::mat4 shortCubeTMat = glm::translate(glm::vec3(10, 6, -10)) * glm::rotate(glm::radians(45.0f), glm::vec3(0, 1, 0)) * glm::scale(glm::vec3(6, 8, 6));
  addMesh("cube.obj", grayDiffuse, shortCubeTMat);

  glm::mat4 lightTMat = floorTMat = glm::translate(glm::vec3(-15.0 / 2.0, 44.9, -15.0 / 2.0)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::vec3(15, 15, 1));
  addMesh("square.obj", whiteLight, lightTMat);

  Frustum frust(aspectRatio, 75.0f, 1.0f);
  glm::vec3 pos(0, 20, -45);
  glm::mat4 rot = Camera::lookAt(pos, glm::vec3(0, 15, 0), glm::vec3(0, 1, 0));
  cam = new Camera(pos, rot, frust);
  background = glm::vec3(0, 0, 0);
}


void Scene::populateCornellBoxDialectric() {
  int redDiffuse = addMat(makeLambertian(glm::vec3(1.0, 0.6, 0.6)));
  int grayDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 0.6)));
  int blueDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 1.0)));
  int whiteLight = addMat(makeLight(glm::vec3(10.0, 10.0, 10.0)));

  int grayMetal = addMat(makeMetal(glm::vec3(0.6f, 0.6f, 0.6f)));

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
  addMesh("cube.obj", grayMetal, tallCubeTMat);

  int glass = addMat(makeDialectric(1.5f));
  addSphere(Sphere(glm::vec3(10, 12, -10), 8.f, glass));

  glm::mat4 lightTMat = floorTMat = glm::translate(glm::vec3(-15.0 / 2.0, 44.9, -15.0 / 2.0)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::vec3(15, 15, 1));
  addMesh("square.obj", whiteLight, lightTMat);

  Frustum frust(aspectRatio, 75.0f, 1.0f);
  glm::vec3 pos(0, 20, -45);
  glm::mat4 rot = Camera::lookAt(pos, glm::vec3(0, 15, 0), glm::vec3(0, 1, 0));
  cam = new Camera(pos, rot, frust);
  background = glm::vec3(0, 0, 0);
}

void Scene::populateSimpleScene() {
  int ground = addMat(makeLambertian(glm::vec3(0.8f, 0.8f, 0.8f)));
  addSphere(Sphere(glm::vec3(0, -1000.5f, -1.0f), 1000.0f, ground));

  int light = addMat(makeLight(glm::vec3(30.0f, 25.0f, 15.0f)));
  addSphere(Sphere(glm::vec3(-1.5f, 1.5f, 0), 0.3f, light));

  /*
  int light2 = addMat(makeLight(glm::vec3(30, 30, 30)));
  glm::mat4 xform = glm::translate(glm::vec3(0, 3, 0)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
  addMesh("square.obj", light2, xform);*/

  glm::mat4 rot = Camera::lookAt(glm::vec3(0, 2, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  Frustum frust(aspectRatio, 60.0f, 1.0f);
  cam = new Camera(glm::vec3(0, 2, 3), rot, frust);
  background = glm::vec3(0.15f, 0.21f, 0.30f);
}

void Scene::populateBVHTest() {
  int redDiffuse = addMat(makeLambertian(glm::vec3(1.0, 0.6, 0.6)));
  int grayDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 0.6)));
  int blueDiffuse = addMat(makeLambertian(glm::vec3(0.6, 0.6, 1.0)));
  int whiteLight = addMat(makeLight(glm::vec3(10.0, 10.0, 10.0)));

  int grayMetal = addMat(makeMetal(glm::vec3(0.6f, 0.6f, 0.6f)));

  /*
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
  addMesh("cube.obj", grayMetal, tallCubeTMat);

  int glass = addMat(makeDialectric(1.5f));
  glm::mat4 shortCubeTMat = glm::translate(glm::vec3(10, 7.35, -10)) *  glm::rotate(glm::radians(90.f), glm::vec3(0,1,0)) * glm::scale(glm::vec3(26, 26, 26));
  addMesh("dragon.obj", glass, shortCubeTMat);

  glm::mat4 lightTMat = floorTMat = glm::translate(glm::vec3(-15.0 / 2.0, 44.9, -15.0 / 2.0)) * glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::vec3(15, 15, 1));
  addMesh("square.obj", whiteLight, lightTMat);

  Frustum frust(aspectRatio, 75.0f, 1.0f);
  glm::vec3 pos(0, 20, -45);
  glm::mat4 rot = Camera::lookAt(pos, glm::vec3(0, 15, 0), glm::vec3(0, 1, 0));
  cam = new Camera(pos, rot, frust);
  background = glm::vec3(0, 0, 0);*/

  // ~ 34-35 average Samples per second with no other programs running
  int orangeDiffuse = addMat(makeLambertian(glm::vec3(0.7f, 0.5f, 0.2f)));
  addMesh("espeon.obj", orangeDiffuse, glm::mat4(1.f));
  glm::mat4 rot = Camera::lookAt(glm::vec3(-11, 9, 10.7), glm::vec3(0, 2, 0), glm::vec3(0, 1, 0));
  Frustum frust(aspectRatio, 60.0f, 1.0f);
  cam = new Camera(glm::vec3(-11, 9, 10.7), rot, frust);
  background = glm::vec3(1, 1, 1);
}