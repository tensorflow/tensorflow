/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_

#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
class Graph;
class Node;
class OpKernel;
class Tensor;

// Represents a single data edge in a `NodeItem`.
struct EdgeInfo {
  // The node ID of the destination in the containing `GraphView`.
  int dst_id;
  // The index of the output that produces values on this edge.
  int output_slot : 31;
  // true if this is the last info for output_slot in the EdgeInfo list.
  bool is_last : 1;
  // The index of the input that consumes values on this edge.
  int input_slot;
};

// Represents a single control edge in a `NodeItem`.
struct ControlEdgeInfo {
  // The node ID of the destination in the containing `GraphView`.
  int dst_id;
};

// Compact structure representing a graph node and its associated kernel.
//
// Each NodeItem is an element of exactly one GraphView.
struct NodeItem {
  // The index of this node's item in its GraphView.
  int node_id = -1;

  // Cached attributes of this node for fast lookup.
  bool kernel_is_async : 1;     // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;            // True iff IsMerge(node)
  bool is_enter : 1;            // True iff IsEnter(node)
  bool is_constant_enter : 1;   // True iff IsEnter(node) and
                                // node->GetAttr("is_constant") == true.
  bool is_exit : 1;             // True iff IsExit(node)
  bool is_control_trigger : 1;  // True iff IsControlTrigger(node)
  bool is_source : 1;           // True iff IsSource(node)
  // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
  bool is_enter_exit_or_next_iter : 1;
  bool is_transfer_node : 1;      // True iff IsTransferNode(node)
  bool is_initialization_op : 1;  // True iff IsInitializationOp(node)
  bool is_recv_or_switch : 1;     // True iff IsRecv(node) || IsSwitch(node)
  bool is_next_iteration : 1;     // True iff IsNextIteration(node)
  bool is_noop : 1;  // True iff item->kernel->type_string_view() == "NoOp")
  bool
      is_any_consumer_merge_or_control_trigger : 1;  // True iff the destination
                                                     // of any output edge is a
                                                     // merge or control trigger
                                                     // node.
  bool is_any_input_ref_typed : 1;  // True iff any IsRefType(dt) for dt in this
                                    // node's input types.
  bool is_distributed_communication : 1;  // True iff the op is registered to
                                          // use distributed communication.
  bool is_send_to_gpu : 1;  // True iff IsSend(node) and the receive device is
                            // GPU.

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  // If the kernel is a Const op, this containts points to the constant tensor.
  const Tensor* const_tensor = nullptr;

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // Number of output edges, excluding control edges.
  int32 num_output_edges;

  // Number of output control edges.
  int32 num_output_control_edges;

  // If non-null, contains an array of num_outputs bools, where the ith bool
  // is true if and only if the ith output is consumed by another node.
  std::unique_ptr<bool[]> outputs_required;

  gtl::MutableArraySlice<EdgeInfo> mutable_output_edges() {
    return gtl::MutableArraySlice<EdgeInfo>(output_edge_base(),
                                            num_output_edges);
  }

  gtl::ArraySlice<EdgeInfo> output_edges() const {
    return gtl::ArraySlice<EdgeInfo>(output_edge_base(), num_output_edges);
  }

  gtl::ArraySlice<ControlEdgeInfo> output_control_edges() const {
    return gtl::ArraySlice<const ControlEdgeInfo>(output_control_edge_base(),
                                                  num_output_control_edges);
  }

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return static_cast<DataType>(input_type_base()[i]);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return static_cast<DataType>(output_type_base()[i]);
  }

  // Return array of per-output allocator attributes.
  const AllocatorAttributes* output_attrs() const { return output_attr_base(); }

  // Return array of expected input index from which each output should
  // be forwarded:
  // kNeverForward (-2) for DO NOT FORWARD (must allocate).
  // kNoReservation (-1) for no expected forwarding.
  // 0... for forward from that input.
  const int* forward_from() const { return forward_from_base(); }

  string DebugString() const;

 private:
  friend class GraphView;

  NodeItem() {}

  // Variable length section starts immediately after *this
  // (uint8 is enough for DataType).
  //   EdgeInfo            out_edges[num_output_edges];
  //   ControlEdgeInfo     out_control_edges[num_output_control_edges];
  //   AllocatorAttributes output_attr[num_outputs];
  //   int                 forward_from[num_outputs];
  //   uint8               input_type[num_inputs];
  //   uint8               output_type[num_outputs];

  // Return pointer to variable length section.
  char* var() const {
    return const_cast<char*>(reinterpret_cast<const char*>(this) +
                             sizeof(NodeItem));
  }

  EdgeInfo* output_edge_base() const {
    return reinterpret_cast<EdgeInfo*>(var());
  }

  ControlEdgeInfo* output_control_edge_base() const {
    return reinterpret_cast<ControlEdgeInfo*>(var() + sizeof(EdgeInfo) *
                                                          num_output_edges);
  }

  AllocatorAttributes* output_attr_base() const {
    return reinterpret_cast<AllocatorAttributes*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges);
  }
  int* forward_from_base() const {
    return reinterpret_cast<int*>(var() + sizeof(EdgeInfo) * num_output_edges +
                                  sizeof(ControlEdgeInfo) *
                                      num_output_control_edges +
                                  sizeof(AllocatorAttributes) * num_outputs);
  }
  uint8* input_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs);
  }
  uint8* output_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs +
        sizeof(uint8) * num_inputs);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};

// Immutable view of a Graph organized for efficient execution.
//
// TODO(b/152651962): Add independent unit tests for this class.
class GraphView {
 public:
  GraphView() : space_(nullptr) {}
  ~GraphView();

  Status Initialize(const Graph* g);
  Status SetAllocAttrs(const Graph* g, const Device* device);
  void SetScopedAllocatorAttrs(const std::vector<const Node*>& sa_nodes);

  // Returns a mutable pointer to the `NodeItem` with the given `id` if it
  // exists in the graph, or `nullptr` if it does not.
  NodeItem* node(int32_t id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    return ((offset == kuint32max)
                ? nullptr
                : reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));
  }

  // Returns the `NodeItem` with the given `id`.
  //
  // REQUIRES: `id` must be the ID of a valid node in the graph.
  const NodeItem& node_ref(int32_t id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    DCHECK_NE(offset, kuint32max);
    return *reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]);
  }

  int32 num_nodes() const { return num_nodes_; }

 private:
  char* InitializeNode(char* ptr, const Node* n);
  size_t NodeItemBytes(const Node* n);

  int32 num_nodes_ = 0;
  uint32* node_offsets_ = nullptr;  // array of size "num_nodes_"
  // node_offsets_[id] holds the byte offset for node w/ "id" in space_

  char* space_;  // NodeItem objects are allocated here

  TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_
