#include "mesh.cuh"

#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <exception>


glm::vec3 grabTriple(std::vector<tinyobj::real_t>& arr, size_t idx) {
  return glm::vec3(arr[3 * idx + 0], arr[3 * idx + 1], arr[3 * idx + 2]);
}

Mesh::Mesh(std::string fpath, std::vector<Triangle>& tris) {
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig readerConfig;
  if (!reader.ParseFromFile(fpath, readerConfig)) {
    std::cerr << "Encountered an error loading file " << fpath << "\n";
    if (!reader.Error().empty()) {
      std::cerr << reader.Error();
    }
    throw std::exception();
  }

  if (!reader.Warning().empty()) {
    std::cerr << "Encountered a warning loading file " << fpath << "\n";
    std::cerr << reader.Warning();
  }

  triOffset = tris.size();

  tinyobj::attrib_t attrib = reader.GetAttrib();
  std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

  for (const auto& shape : shapes) {
    size_t index_offset = 0;

    // faces in shape
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shape.mesh.num_face_vertices[f]);

      if (fv != 3) {
        std::cerr << "Mesh loaded from " << fpath << " contains non-triangular face!\n";
        return;
      }

      tinyobj::index_t idx0 = shape.mesh.indices[index_offset];
      tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
      tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
      glm::vec3 v0 = grabTriple(attrib.vertices, idx0.vertex_index);
      glm::vec3 v1 = grabTriple(attrib.vertices, idx1.vertex_index);
      glm::vec3 v2 = grabTriple(attrib.vertices, idx2.vertex_index);

      glm::vec3 n0, n1, n2;
      // No normal data available
      if (idx0.normal_index < 0 || idx1.normal_index < 0 || idx2.normal_index < 0) {
        n0 = n1 = n2 = glm::cross(v1 - v0, v2 - v0);
      }
      else {
        n0 = grabTriple(attrib.normals, idx0.normal_index);
        n1 = grabTriple(attrib.normals, idx1.normal_index);
        n2 = grabTriple(attrib.normals, idx2.normal_index);
      }

      Triangle tri = Triangle(Vertex(v0, n0), Vertex(v1, n1), Vertex(v2, n2));
      tris.push_back(tri);

      index_offset += fv;
    }
  }

  n_tris = tris.size() - triOffset;
}

MeshInstance::MeshInstance(Mesh& base, int matIdx, glm::mat4 tMat) : matIdx(matIdx), tMat(tMat) {
  triOffset = base.triOffset;
  n_tris = base.n_tris;

  tMatInv = glm::inverse(tMat);
  tMatInvTpose = glm::transpose(tMatInv);
}

CUDA_CALLABLE_MEMBER void MeshInstance::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, Triangle* tris) {
  Ray local = Ray(glm::vec3(tMatInv * glm::vec4(r.origin, 1.0)),
                  glm::vec3(tMatInv * glm::vec4(r.dir, 0.0)));

  HitRecord temp;
  temp.isHit = rec.isHit = false;
  float closest = t_max;
  for (int i = 0; i < n_tris; i++) {
    tris[triOffset + i].hit(local, t_min, t_max, temp);
    if (temp.isHit && temp.t < closest) {
      closest = temp.t;
      rec = temp;
    }
  }

  // Re-adjust normal and intersection points to world coordinates
  if (rec.isHit) {
    //rec.setNormal(r, glm::normalize(glm::vec3(tMatInvTpose * glm::vec4(rec.normal, 0.0f))));
    //rec.point = glm::vec3(tMat * glm::vec4(rec.point, 1.0f));
  }

}