#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <unordered_map>

#include "hittable.cuh"
#include "material.cuh"

class Scene {
  std::string resourceDir;

  // Loads the mesh at the given filepath. Returns null if mesh could not be loaded
  // Filepath should be relative to resourceDir
  //Mesh* loadMesh(std::string fpath);

public:
  std::vector<Sphere*> sphere;
  std::vector<Material*> materials;
  //std::unordered_map<std::string, Mesh*> meshes;


  Scene();
  Scene(std::string resourceDir) : resourceDir(resourceDir) {};

  // When adding an object or mat, you give full 
  // ownership of the data to the Scene class
  Hittable* addSphere(Sphere* sph);
  int addMat(Material* mat); // Returns the material ID


  void freeAll();
};


#endif