#include "mesh.cuh"

#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <exception>
#include <algorithm>

#include "aabb.cuh"
#include "sah_bvh.h"


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

int Mesh::initAcceleration(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris, int start, int end) {
  int span = end - start;

  int axis = rand() % 3;
  auto comparator = axis == 0 ? box_compare_x
                  : axis == 1 ? box_compare_y
                  : box_compare_z;

  AABB l, r;
  if (span == 1) {
    // Only a single tri, no need to do any merging
    tris[start].bbox(l);
    BVHNode left = BVH::makeLeaf(l, start, end);
    nodes.push_back(left);
    return int(nodes.size()) - 1;
  } /*else if (span == 2) {
    tris[start].bbox(l);
    tris[start + 1].bbox(r);

    
    // Make a single node containing the two tris
    BVHNode node = BVH::makeLeaf(AABB(l, r), start, end);
    nodes.push_back(node);
    return int(nodes.size()) - 1;

  }*/ else {
    std::sort(tris.begin() + start, tris.begin() + end, comparator);

    // reserve space for current with a blank node
    nodes.push_back(BVHNode());
    int idx = int(nodes.size()) - 1;

    int mid = start + span / 2;
    int left = initAcceleration(nodes, tris, start, mid);
    int right = initAcceleration(nodes, tris, mid, end);

    AABB merged = AABB(BVH::getAABB(&nodes[0], left), BVH::getAABB(&nodes[0], right));
    nodes[idx] = BVH::makeInternal(merged, left, right);

    return idx;
  }
}

void Mesh::initAcceleration(std::vector<BVHNode>& nodes, std::vector<Triangle>& tris) {
  // assign offset in global array
  //bvhOffset = initAcceleration(nodes, tris, triOffset, triOffset + n_tris);

  bvhOffset = SAH_BVH::createBVH(nodes, tris, triOffset, triOffset + n_tris);
}




MeshInstance::MeshInstance(Mesh& base, int matIdx, glm::mat4 tMat) : matIdx(matIdx), tMat(tMat) {
  triOffset = base.triOffset;
  n_tris = base.n_tris;
  bvhOffset = base.bvhOffset;

  tMatInv = glm::inverse(tMat);
  tMatInvTpose = glm::transpose(tMatInv);
}

CUDA_CALLABLE_MEMBER bool MeshInstance::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, BVHNode* nodes, Triangle* tris) {
  Ray local = Ray(glm::vec3(tMatInv * glm::vec4(r.origin, 1.0)),
                  glm::vec3(tMatInv * glm::vec4(r.dir, 0.0)));

  /*
  HitRecord temp;
  bool hit = false;
  rec.t = t_max;
  for (int i = 0; i < n_tris; i++) {
    if (tris[triOffset + i].hit(local, t_min, t_max, temp) && temp.t < rec.t) {
      rec = temp;
      hit = true;
    }
  }*/

  bool hit = BVH::traverseIterative(nodes, tris, local, t_min, t_max, rec, bvhOffset);

  // Re-adjust normal and intersection points to world coordinates
  if (hit) {
    rec.setNormal(r, glm::normalize(glm::vec3(tMatInvTpose * glm::vec4(rec.normal, 0.0f))));
    //rec.normal = glm::normalize(glm::vec3(tMatInvTpose * glm::vec4(rec.normal, 0.0f)));
    //rec.point = glm::vec3(tMat * glm::vec4(local.at(rec.t), 1.0f));
  }
  return hit;
}