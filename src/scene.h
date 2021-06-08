#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <unordered_map>

#include "hittable.cuh"
#include "material.cuh"
#include "mesh.cuh"

struct PTData;

class Scene {
  std::string resourceDir;

  // Loads the mesh at the given filepath. Returns null if mesh could not be loaded
  // Filepath should be relative to resourceDir
  Mesh loadMesh(std::string fpath);

public:
  // All to be sent to GPU
  std::vector<Sphere> spheres;
  std::vector<Material> materials;
  std::vector<Triangle> tris;
  std::vector<MeshInstance> instances;

  // Book keeping on CPU
  std::unordered_map<std::string, Mesh> meshes;


  Scene() : spheres(), materials(), tris(), instances(), resourceDir("") {};
  Scene(std::string resourceDir) : resourceDir(resourceDir) {};

  // When adding an object or mat, you give full 
  // ownership of the data to the Scene class
  void addSphere(Sphere& sph);
  int addMat(Material& mat); // Returns the material ID
  void addMesh(std::string fpath, int matIdx, glm::mat4 tMat);

  void copyToGPU(PTData& data);
  void freeFromGPU(PTData& data);


  void freeAll();



  void populateRandomScene();
  void populateComplexMesh();
  void populateCornellBox();
};


#endif