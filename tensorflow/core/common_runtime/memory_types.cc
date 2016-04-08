/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/common_runtime/memory_types.h"

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Status ValidateMemoryTypes(DeviceType device_type, const Graph* g) {
  if (device_type != DEVICE_GPU) {
    // On non-GPU devices, HOST_MEMORY and DEVICE_MEMORY are always
    // compatible.
    return Status::OK();
  }
  // For GPU device, HOST_MEMORY and DEVICE_MEMORY is not
  // compatible. I.e., a conversion/transfer must be done.
  struct Endpoint {
    int id;
    int off;
  };
  struct EndpointHash {
    uint32 operator()(const Endpoint& x) const {
      return Hash32(reinterpret_cast<const char*>(&x.id), sizeof(int), x.off);
    }
  };
  struct EndpointEq {
    uint32 operator()(const Endpoint& x, const Endpoint& y) const {
      return (x.id == y.id) && (x.off == y.off);
    }
  };
  // {node id, slot id} -> memory type.
  typedef std::unordered_map<Endpoint, MemoryType, EndpointHash, EndpointEq>
      MemTypeMap;
  MemTypeMap inp;
  MemTypeMap out;
  MemoryTypeVector inp_mvec;
  MemoryTypeVector out_mvec;
  for (const Node* n : g->nodes()) {
    TF_RETURN_IF_ERROR(MemoryTypesForNode(g->op_registry(), device_type,
                                          n->def(), &inp_mvec, &out_mvec));
    for (size_t i = 0; i < inp_mvec.size(); ++i) {
      inp[{n->id(), static_cast<int>(i)}] = inp_mvec[i];
    }
    for (size_t i = 0; i < out_mvec.size(); ++i) {
      out[{n->id(), static_cast<int>(i)}] = out_mvec[i];
    }
  }
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    MemoryType sm = gtl::FindWithDefault(out, {e->src()->id(), e->src_output()},
                                         DEVICE_MEMORY);
    MemoryType dm = gtl::FindWithDefault(inp, {e->dst()->id(), e->dst_input()},
                                         DEVICE_MEMORY);
    VLOG(1) << e->src()->id() << ":" << e->src_output() << " -> "
            << e->dst()->id() << ":" << e->dst_input() << ": " << sm << " -> "
            << dm;
    if (sm != dm) {
      return errors::Internal(
          "Memory type mismatch (", sm, " ", dm, ") between :", e->src()->id(),
          ":", e->src_output(), " and ", e->dst()->id(), ":", e->dst_input(),
          " : from ", e->src()->DebugString(), " to ", e->dst()->DebugString());
    }
  }
  return Status::OK();
}

Status MemoryTypeForOutput(DeviceType device_type, const Graph* g,
                           const Node* n, int index, MemoryType* memory_type) {
  MemoryTypeVector inp_mvec;
  MemoryTypeVector out_mvec;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(g->op_registry(), device_type, n->def(),
                                        &inp_mvec, &out_mvec));
  if (out_mvec.size() <= index) {
    return errors::Internal("Trying to get the memory type for ", index,
                            "'th output of node ", n->DebugString(),
                            " that has only ", out_mvec.size(), " outputs");
  }
  *memory_type = out_mvec[index];
  return Status::OK();
}

}  // end namespace tensorflow
