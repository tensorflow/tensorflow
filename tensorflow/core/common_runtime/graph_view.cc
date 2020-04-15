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

#include "tensorflow/core/common_runtime/graph_view.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

string NodeItem::DebugString() const {
  string ret = strings::StrCat("{name:'", kernel->name(), "' id:", node_id);
  if (is_source) {
    strings::StrAppend(&ret, " source}");
  } else {
    strings::StrAppend(&ret, " def:{", SummarizeNodeDef(kernel->def()), "}}");
  }
  return ret;
}

GraphView::~GraphView() {
  static_assert(std::is_trivially_destructible<AllocatorAttributes>::value,
                "Update code if AllocatorAttributes gains a destructor");
  static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                "Update code if EdgeInfo gains a destructor");
  for (int i = 0; i < num_nodes_; i++) {
    NodeItem* n = node(i);
    if (n != nullptr) {
      n->NodeItem::~NodeItem();
      // Memory for "n" itself is held in space_ & gets cleaned up below
    }
  }
  delete[] node_offsets_;
  delete[] space_;
}

namespace {
typedef std::tuple<int32, int32> OutputAndControlEdges;

OutputAndControlEdges CountOutputEdges(const Node* n) {
  DCHECK_LE(n->out_edges().size(), kint32max);
  int32 num_output_edges = 0;
  int32 num_output_control_edges = 0;
  for (auto e : n->out_edges()) {
    if (IsSink(e->dst())) continue;
    if (e->IsControlEdge()) {
      ++num_output_control_edges;
    } else {
      ++num_output_edges;
    }
  }
  return OutputAndControlEdges(num_output_edges, num_output_control_edges);
}
}  // namespace

size_t GraphView::NodeItemBytes(const Node* n) {
  int32 num_output_edges;
  int32 num_output_control_edges;
  std::tie(num_output_edges, num_output_control_edges) = CountOutputEdges(n);
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  // Compute number of bytes needed for NodeItem and variable length data.
  // We do not subtract sizeof(var) since num_inputs/num_outputs might
  // both be zero.
  const size_t raw_bytes =
      sizeof(NodeItem)                             // Fixed
      + num_output_edges * sizeof(EdgeInfo)        // output_edges[...]
      + num_output_control_edges *                 //
            sizeof(ControlEdgeInfo)                // output_control_edges[...]
      + num_outputs * sizeof(AllocatorAttributes)  // output_attr[...]
      + num_outputs * sizeof(int)                  // forward_from[num_outputs]
      + num_inputs * sizeof(uint8)                 // input_type[num_inputs]
      + num_outputs * sizeof(uint8);               // output_type[num_outputs]
  static constexpr size_t kItemAlignment = sizeof(NodeItem*);
  static_assert(kItemAlignment % alignof(NodeItem) == 0,
                "NodeItem must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(EdgeInfo) == 0,
                "EdgeInfo must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(ControlEdgeInfo) == 0,
                "ControlEdgeInfo must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(AllocatorAttributes) == 0,
                "AllocatorAttributes must be aligned with kItemAlignment");
  static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0,
                "NodeItem must be aligned with EdgeInfo");
  static_assert(sizeof(NodeItem) % alignof(AllocatorAttributes) == 0,
                "NodeItem must be aligned with AllocatorAttributes");
  static_assert(sizeof(EdgeInfo) % alignof(AllocatorAttributes) == 0,
                "EdgeInfo must be aligned with AllocatorAttributes");
  const size_t bytes =
      ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
  return bytes;
}

char* GraphView::InitializeNode(char* ptr, const Node* n) {
  const int id = n->id();
  CHECK(node_offsets_[id] == kuint32max);  // Initial value in constructor

  const size_t bytes = NodeItemBytes(n);
  constexpr size_t kItemAlignment = sizeof(NodeItem*);
  CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
  NodeItem* item = reinterpret_cast<NodeItem*>(ptr);

  // We store a 32-bit offset relative to the beginning of space_, so that we
  // only need an array of 32-bit values to map from node id to the NodeItem*,
  // (versus 64 bits on most machines if we just stored an array of NodeItem*
  // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
  // values as "int" vs "size_t" in CHECK_LE.
  CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
  const uint32 offset = static_cast<uint32>(ptr - space_);
  node_offsets_[id] = offset;
  ptr += bytes;

  int32 num_output_edges;
  int32 num_output_control_edges;
  std::tie(num_output_edges, num_output_control_edges) = CountOutputEdges(n);
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  new (item) NodeItem();
  item->num_inputs = num_inputs;
  item->num_outputs = num_outputs;
  item->num_output_edges = num_output_edges;
  item->num_output_control_edges = num_output_control_edges;

  // Fill output edges.
  // Keep track of the last EdgeInfo in the EdgeInfo array that references
  // a given output slot.  For all but the last, we need to do a copy of the
  // Tensor when propagating results downstream in the graph, but for the
  // last one, we can just do a move of the Tensor object to propagate it.
  gtl::InlinedVector<EdgeInfo*, 4> last_indices(num_outputs, nullptr);
  EdgeInfo* dst_edge = item->output_edge_base();
  for (auto e : n->out_edges()) {
    if (e->IsControlEdge()) continue;
    dst_edge->dst_id = e->dst()->id();
    CHECK_LE(e->src_output(), 0x3FFFFFFF);  // Must fit in 31 bits
    dst_edge->output_slot = e->src_output();
    dst_edge->is_last = false;
    const int output_slot = dst_edge->output_slot;
    if (output_slot >= 0) {
      last_indices[output_slot] = dst_edge;
    }
    // NOTE: The `input_slot` will be rewritten to the frame-wide offset later
    // in `ExecutorImpl::Initialize()`.
    dst_edge->input_slot = e->dst_input();
    dst_edge++;
  }
  for (EdgeInfo* edge_info : last_indices) {
    if (edge_info != nullptr) {
      edge_info->is_last = true;
    }
  }
  ControlEdgeInfo* dst_control_edge = item->output_control_edge_base();
  for (auto e : n->out_edges()) {
    if (!e->IsControlEdge() || IsSink(e->dst())) continue;
    dst_control_edge->dst_id = e->dst()->id();
    dst_control_edge++;
  }

  AllocatorAttributes* output_attrs = item->output_attr_base();
  for (int i = 0; i < num_outputs; i++) {
    new (&output_attrs[i]) AllocatorAttributes();
  }

  DCHECK_LT(DataType_MAX, 255);  // Must fit in uint8
  uint8* input_types = item->input_type_base();
  for (int i = 0; i < num_inputs; i++) {
    input_types[i] = static_cast<uint8>(n->input_type(i));
    DCHECK_EQ(item->input_type(i), n->input_type(i));
  }

  // Check ScopedAllocatorAttrs and forward_from.  Also assign output_types.
  {
    std::vector<int> forward_input;
    Status fwd_status =
        GetNodeAttr(n->attrs(), "_forward_input", &forward_input);
    std::vector<int> scoped_allocator_attrs;
    Status sa_status =
        GetNodeAttr(n->attrs(), "_scoped_allocator", &scoped_allocator_attrs);

    int* forward_from = item->forward_from_base();
    uint8* output_types = item->output_type_base();
    for (int i = 0; i < num_outputs; ++i) {
      output_types[i] = static_cast<uint8>(n->output_type(i));
      DCHECK_EQ(item->output_type(i), n->output_type(i));

      forward_from[i] = OpKernelContext::Params::kNoReservation;
      if (sa_status.ok()) {
        for (int j = 0; j < scoped_allocator_attrs.size(); j += 2) {
          if (scoped_allocator_attrs[j] == i) {
            // This output slot must be explicitly allocated from a
            // ScopedAllocator.
            forward_from[i] = OpKernelContext::Params::kNeverForward;
            DCHECK_EQ(output_attrs[i].scope_id, 0);
            output_attrs[i].scope_id = scoped_allocator_attrs[j + 1];
          }
        }
      }
      if (fwd_status.ok() &&
          forward_from[i] == OpKernelContext::Params::kNoReservation) {
        DCHECK_EQ(forward_input.size() % 2, 0);
        for (int j = 0; j < forward_input.size(); j += 2) {
          if (forward_input[j + 1] == i) {
            DCHECK_EQ(forward_from[i], OpKernelContext::Params::kNoReservation);
            forward_from[i] = forward_input[j];
            break;
          }
        }
      }
    }
  }

  return ptr;
}

Status GraphView::Initialize(const Graph* g) {
  CHECK(node_offsets_ == nullptr);
  const int num_nodes = g->num_node_ids();
  num_nodes_ = num_nodes;
  size_t total_bytes = 0;
  for (const Node* n : g->nodes()) {
    if (n->out_edges().size() > kint32max) {
      return errors::InvalidArgument(
          "The executor cannot handle nodes with more than ", kint32max,
          " output edges. Node ", n->name(), " had ", n->out_edges().size(),
          " output edges.");
    }
    total_bytes += NodeItemBytes(n);
  }

  node_offsets_ = new uint32[num_nodes];
  for (int i = 0; i < num_nodes; i++) {
    node_offsets_[i] = kuint32max;
  }

  space_ = new char[total_bytes];  // NodeItem objects are allocated here
  char* ptr = space_;
  for (const Node* n : g->nodes()) {
    ptr = InitializeNode(ptr, n);
  }
  CHECK_EQ(ptr, space_ + total_bytes);
  return Status::OK();
}

namespace {
// If a Node has been marked to use a ScopedAllocator x for output i, then
// sc_attr will contain the subsequence (i, x) at an even offset.  This function
// extracts and transfers that ScopedAllocator id to alloc_attr.  For now, we
// only allow one ScopedAllocator use per Node.
bool ExtractScopedAllocatorAttr(const std::vector<int>& sc_attr,
                                int output_index,
                                AllocatorAttributes* alloc_attr) {
  DCHECK_LE(2, sc_attr.size());
  for (int i = 0; i < sc_attr.size(); i += 2) {
    if (sc_attr[i] == output_index) {
      CHECK_EQ(alloc_attr->scope_id, 0);
      alloc_attr->scope_id = sc_attr[i + 1];
      return true;
    }
  }
  return false;
}
}  // namespace

void GraphView::SetScopedAllocatorAttrs(
    const std::vector<const Node*>& sa_nodes) {
  for (const Node* sa : sa_nodes) {
    NodeItem* sa_item = node(sa->id());
    AllocatorAttributes* sa_attrs = sa_item->output_attr_base();
    // Control edges out of the ScopedAllocator should be use instances, but may
    // include a few other nodes.
    for (const auto& e : sa->out_edges()) {
      if (IsSink(e->dst()) || !e->IsControlEdge()) {
        continue;
      }
      Node* use_node = e->dst();
      NodeItem* item = node(use_node->id());
      AllocatorAttributes* use_attrs = item->output_attr_base();
      std::vector<int> scoped_allocator_attrs;
      Status s = GetNodeAttr(use_node->attrs(), "_scoped_allocator",
                             &scoped_allocator_attrs);
      if (!s.ok()) {
        VLOG(2) << "Failed to find expected ScopedAllocator attr on "
                << use_node->name();
        continue;
      }
      // There can be more than one output using ScopedAllocation, but this
      // analysis assumes they use the same ScopedAllocator.
      for (const auto& e : use_node->out_edges()) {
        if (IsSink(e->dst()) || !e->IsControlEdge()) {
          AllocatorAttributes attr;
          if (ExtractScopedAllocatorAttr(scoped_allocator_attrs,
                                         e->src_output(), &attr)) {
            // Set the scope_id on this use instance node.
            (use_attrs + e->src_output())->Merge(attr);
            // Propagate the other attributes of this node back to the SA node.
            attr = *(use_attrs + e->src_output());
            attr.scope_id = 0;
            sa_attrs->Merge(attr);
          }
        }
      }
    }
  }
}

namespace {
Status InferAllocAttr(const Node* n, const Node* dst,
                      const DeviceNameUtils::ParsedName& local_dev_name,
                      AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->attrs(), "send_device", &src_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_src_name;
    if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
      s = errors::Internal("Bad send_device attr '", src_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
      // Value is going to be the sink of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
    } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) &&
               parsed_src_name.type != "CPU") {
      // Value is going to be the sink of a local DMA from GPU to CPU (or
      // other types of accelerators).
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_src_name.type;
    }
  }
  if (IsSend(dst)) {
    string dst_name;
    s = GetNodeAttr(dst->attrs(), "recv_device", &dst_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_dst_name;
    if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
      s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
      // Value is going to be the source of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of an RPC out";
    } else if ((local_dev_name.type == "CPU" || dst->IsHostSend()) &&
               parsed_dst_name.type != "CPU") {
      // Value is going to be the source of a local DMA from CPU to GPU (or
      // other types of accelerators).
      // Note that this does not cover the case where the allocation of the
      // output tensor is not generated by the src: n.
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_dst_name.type;
    }
  }
  if (n->IsCollective()) {
    // We'll make the sweeping assumption that any collective op is going
    // to be involved in network i/o.
    attr->set_nic_compatible(true);
  }
  return s;
}
}  // namespace

Status GraphView::SetAllocAttrs(const Graph* g, const Device* device) {
  Status s;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  std::vector<const Node*> scoped_allocator_instances;
  for (const Node* n : g->nodes()) {
    NodeItem* item = node(n->id());
    AllocatorAttributes* attrs = item->output_attr_base();
    if (IsScopedAllocator(n)) {
      scoped_allocator_instances.push_back(n);
    }

    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (const auto& e : n->out_edges()) {
      if (!e->IsControlEdge()) {
        AllocatorAttributes attr;
        s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
        if (!s.ok()) return s;
        if (attr.value != 0 || attr.scope_id != 0) {
          attrs[e->src_output()].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      const OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      if (on_host) {
        AllocatorAttributes h;
        h.set_on_host(on_host);
        attrs[out].Merge(h);
      }
    }
  }
  SetScopedAllocatorAttrs(scoped_allocator_instances);
  return s;
}

}  // namespace tensorflow
