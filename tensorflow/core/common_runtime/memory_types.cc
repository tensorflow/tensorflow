/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

struct Endpoint {
  int node_id;
  int output_index;
};

struct EndpointHash {
  uint32 operator()(const Endpoint& x) const {
    return Hash32(reinterpret_cast<const char*>(&x.node_id), sizeof(int),
                  x.output_index);
  }
};

struct EndpointEq {
  uint32 operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node_id == y.node_id) && (x.output_index == y.output_index);
  }
};

static Status ProcessMemoryTypes(
    const DeviceType& device_type, const Graph* g,
    const std::function<Status(const Edge*, MemoryType, MemoryType)>& fn) {
  if (device_type != DEVICE_GPU &&
      !DeviceFactory::IsPluggableDevice(device_type.type_string())) {
    // On non-GPU devices, HOST_MEMORY and DEVICE_MEMORY are always compatible.
    return Status::OK();
  }
  // For GPU, HOST_MEMORY and DEVICE_MEMORY is not compatible. I.e., a
  // conversion/transfer must be done.
  //
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
      VLOG(2) << "inp mvec " << n->id() << " " << i << " " << inp_mvec[i];
      inp[{n->id(), static_cast<int>(i)}] = inp_mvec[i];
    }
    for (size_t i = 0; i < out_mvec.size(); ++i) {
      VLOG(2) << "out mvec " << n->id() << " " << i << " " << out_mvec[i];
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
    TF_RETURN_IF_ERROR(fn(e, sm, dm));
  }
  return Status::OK();
}

Status ValidateMemoryTypes(const DeviceType& device_type, const Graph* g) {
  return ProcessMemoryTypes(
      device_type, g, [](const Edge* e, MemoryType sm, MemoryType dm) {
        if (sm == dm) {
          return Status::OK();
        }
        return errors::Internal("Memory type mismatch (", sm, " ", dm,
                                ") between :", e->src()->id(), ":",
                                e->src_output(), " and ", e->dst()->id(), ":",
                                e->dst_input(), " : from ",
                                FormatNodeForError(*e->src()), " to ",
                                FormatNodeForError(*e->dst()));
      });
}

// Given an Edge whose two endpoints have different memory types and
// are gonna to insert a pair of HostSend/Recv or Send/HostRecv nodes,
// GetTensorName() returns a unique string that we can use as part of
// the rendezvous key. The return string is guaranteed to be unique
// within this process. That is sufficient because EnsureMemoryTypes
// is only used on a TensorFlow graph that is gonna to be executed in
// a single tf device (hence within a single process).
static string GetTensorName(const Edge* edge) {
  static std::atomic<int64> counter(0);
  return strings::StrCat("memtype_", counter.fetch_add(1), "_",
                         edge->src()->name());
}

static Node* Send(Graph* g, const string& tensor_name,
                  const string& device_name, bool host, const Edge* edge) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), host ? "_HostSend" : "_Send")
                  .Input(edge->src(), edge->src_output())
                  .Attr("tensor_name", tensor_name)
                  .Attr("send_device", device_name)
                  .Attr("send_device_incarnation", 0)  // Do not care.
                  .Attr("recv_device", device_name)
                  .Attr("_hostmem_sendrecv", true)
                  .Attr("_src", edge->src()->name())
                  .Attr("_dst", edge->dst()->name())
                  .Finalize(g, &ret));
  return ret;
}

static Node* Recv(Graph* g, const string& tensor_name,
                  const string& device_name, bool host, const Edge* edge) {
  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), host ? "_HostRecv" : "_Recv")
          .Attr("tensor_type", edge->src()->output_type(edge->src_output()))
          .Attr("tensor_name", tensor_name)
          .Attr("send_device", device_name)
          .Attr("send_device_incarnation", 0)
          .Attr("recv_device", device_name)
          .Attr("_hostmem_sendrecv", true)
          .Attr("_src", edge->src()->name())
          .Attr("_dst", edge->dst()->name())
          .Finalize(g, &ret));
  return ret;
}

Status EnsureMemoryTypes(const DeviceType& device_type,
                         const string& device_name, Graph* g) {
  struct Item {
    const Edge* edge;
    MemoryType sm;
    MemoryType dm;
  };
  std::vector<Item> edges;
  TF_RETURN_IF_ERROR(ProcessMemoryTypes(
      device_type, g, [&edges](const Edge* e, MemoryType sm, MemoryType dm) {
        if (sm == dm) {
          return Status::OK();
        }
        if (((sm == HOST_MEMORY) && (dm == DEVICE_MEMORY)) ||
            ((sm == DEVICE_MEMORY) && (dm == HOST_MEMORY))) {
          edges.push_back({e, sm, dm});
          return Status::OK();
        }
        return errors::Internal("Unexpected memory type pair on an edge: ", sm,
                                " vs. ", dm);
      }));

  // edges contains edges in 'g' that memtype is not
  // compatible. Therefore, if we found any, we need to insert
  // HostSend/Recv and Send/HostRecv pairs.  recv_nodes records all
  // nodes we added so that we don't copy the same tensor more than
  // once.
  if (!edges.empty()) {
    std::unordered_map<Endpoint, Node*, EndpointHash, EndpointEq> recv_nodes;
    for (const auto& item : edges) {
      const Edge* e = item.edge;
      const bool has_ref = IsRefType(e->src()->output_type(e->src_output()));
      Node* recv = nullptr;
      Endpoint key{e->src()->id(), e->src_output()};
      auto iter = recv_nodes.find(key);
      if (iter == recv_nodes.end()) {
        const string tensor_name = GetTensorName(e);
        Node* send =
            Send(g, tensor_name, device_name, (item.sm == HOST_MEMORY), e);
        recv = Recv(g, tensor_name, device_name, (item.dm == HOST_MEMORY), e);
        if (!has_ref) {
          // We only cache if there is no ref is involved.
          recv_nodes[key] = recv;
        }
        g->AddControlEdge(send, recv);
      } else {
        recv = iter->second;
      }
      g->AddEdge(recv, 0, e->dst(), e->dst_input());
      g->RemoveEdge(e);
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Dumped graph after EnsureMemoryTypes to "
            << DumpGraphToFile("EnsureMemoryTypes", *g);
  }

  return ValidateMemoryTypes(device_type, g);
}

Status MemoryTypeForOutput(const DeviceType& device_type, const Graph* g,
                           const Node* n, int index, MemoryType* memory_type) {
  MemoryTypeVector inp_mvec;
  MemoryTypeVector out_mvec;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(g->op_registry(), device_type, n->def(),
                                        &inp_mvec, &out_mvec));
  if (out_mvec.size() <= index) {
    return errors::Internal("Trying to get the memory type for ", index,
                            "'th output of node ", FormatNodeForError(*n),
                            " that has only ", out_mvec.size(), " outputs");
  }
  *memory_type = out_mvec[index];
  return Status::OK();
}

}  // end namespace tensorflow
