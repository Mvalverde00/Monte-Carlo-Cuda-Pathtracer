#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <unordered_map>

#include "hittable.cuh"
#include "material.cuh"
#include "mesh.cuh"
#include "bvh.cuh"

struct PTData;
class Camera;

class Scene {
  std::string resourceDir;
  float aspectRatio = (16.f/9.f); // Assume a 16/9 aspect ratio by default which is most common.
  glm::vec3 background;

  // Loads the mesh at the given filepath. Returns null if mesh could not be loaded
  // Filepath should be relative to resourceDir
  Mesh loadMesh(std::string fpath);

public:
  // All to be sent to GPU
  std::vector<Sphere> spheres;
  std::vector<Material> materials;
  std::vector<Triangle> tris;
  std::vector<MeshInstance> instances;
  std::vector<BVHNode> nodes;
  Camera* cam;

  // Book keeping on CPU
  std::unordered_map<std::string, Mesh> meshes;


  Scene() : resourceDir("") {};
  Scene(std::string resourceDir) : resourceDir(resourceDir) {};
  Scene(std::string resourceDir, float aR) : resourceDir(resourceDir), aspectRatio(aR) {};

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
  void populateCornellBoxMetal();
  void populateCornellBoxDialectric();

  void populateSimpleScene();
  void populateBVHTest();
};


#endif