/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Compilation for distributed TPU (TPU_REPLICATED_CORE devices).

#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/array2d.h"
#include "xla/array4d.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/service/computation_placer.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_propagation.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/graph_rewrite/cond_builder.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass_internal.h"
#include "tensorflow/core/tpu/graph_rewrite/host_training_loop_optimization_util.h"
#include "tensorflow/core/tpu/graph_rewrite/incomplete_nodedef_builder.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_fingerprint_utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

// Device coordinates are defined as (x, y, z, core), thus resulting in a rank 4
// topology.
constexpr int kTPUTopologyRank = 4;

// An upper bound on how many cores may be present in the topology.
static constexpr int kTPUMaxTopologySize = 4096;

// Attribute containing the serialized xla::OpSharding to be passed to the
// corresponding XLA HLO operation, which represents how a shape is distributed
// across logical cores, e.g., replication, single-device, or partitioning.
const char kShardingAttribute[] = "_XlaSharding";

const char kTPUPartitionedInput[] = "TPUPartitionedInput";
const char kTPUPartitionedInputV2[] = "TPUPartitionedInputV2";

const char kTPUPartitionedOutput[] = "TPUPartitionedOutput";
const char kTPUPartitionedOutputV2[] = "TPUPartitionedOutputV2";

const char kVarHandleOp[] = "VarHandleOp";

static const char* const kTPUCompilationResultAttr = "_tpu_compilation_status";
static const char* const kPostDeviceRewriteAttr = "_post_device_rewrite";

using NodeAndId = std::pair<const Node*, int>;

struct NodeAndPort {
  explicit NodeAndPort(Node* node, int port) : node(node), port(port) {}

  Node* node;
  // Port of the node, e.g. this can be the `src_output` index of an Edge.
  int port;
};

class IntrusiveHeapLink {
 public:
  using size_type = size_t;
  static constexpr size_type kNotMember = std::numeric_limits<size_type>::max();

  IntrusiveHeapLink() = default;

  // Only IntrusiveHeap and LinkAccess objects should make these objects.
  explicit IntrusiveHeapLink(size_type pos) : pos_{pos} {}

  // Only IntrusiveHeap and LinkAccess should get the value.
  size_type get() const { return pos_; }

 private:
  size_type pos_{kNotMember};
};

template <typename T, IntrusiveHeapLink T::*M>
struct IntrusiveHeapDataMemberLinkAccess {
  IntrusiveHeapLink Get(const T* elem) const { return elem->*M; }
  void Set(T* elem, IntrusiveHeapLink link) const { elem->*M = link; }
};

template <typename T>
struct DefaultIntrusiveHeapLinkAccess {
  IntrusiveHeapLink Get(const T* elem) const { return elem->heap; }
  void Set(T* elem, IntrusiveHeapLink link) const { elem->heap = link; }
};

template <typename T, typename PtrCompare,
          typename LinkAccess = DefaultIntrusiveHeapLinkAccess<T>,
          typename Alloc = std::allocator<T*>>
class IntrusiveHeap {
 public:
  typedef typename IntrusiveHeapLink::size_type size_type;
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef PtrCompare pointer_compare_type;
  typedef LinkAccess link_access_type;
  typedef Alloc allocator_type;

  explicit IntrusiveHeap(
      const pointer_compare_type& comp = pointer_compare_type(),
      const link_access_type& link_access = link_access_type(),
      const allocator_type& alloc = allocator_type())
      : rep_(comp, link_access, alloc) {}

  size_type size() const { return heap().size(); }

  bool empty() const { return heap().empty(); }

  // Return the top element, but don't remove it.
  pointer top() const {
    DCHECK(!empty());
    return heap()[0];
  }

  // Remove the top() pointer from the heap and return it.
  pointer Pop() {
    pointer t = top();
    Remove(t);
    return t;
  }

  // Insert 't' into the heap.
  void Push(pointer t) {
    SetPositionOf(t, heap().size());
    heap().push_back(t);
    FixHeapUp(t);
  }

  // Adjust the heap to accommodate changes in '*t'.
  void Adjust(pointer t) {
    DCHECK(Contains(t));
    size_type h = GetPositionOf(t);
    if (h != 0 && compare()(t, heap()[(h - 1) >> 1])) {
      FixHeapUp(t);
    } else {
      FixHeapDown(t);
    }
  }

  // Remove the specified pointer from the heap.
  void Remove(pointer t) {
    DCHECK(Contains(t));
    size_type h = GetPositionOf(t);
    SetPositionOf(t, IntrusiveHeapLink::kNotMember);
    if (h == heap().size() - 1) {
      // Fast path for removing from back of heap.
      heap().pop_back();
      return;
    }
    // Move the element from the back of the heap to overwrite 't'.
    pointer& elem = heap()[h];
    elem = heap().back();
    SetPositionOf(elem, h);  // Element has moved, so update its link.
    heap().pop_back();
    Adjust(elem);  // Restore the heap invariant.
  }

  void Clear() { heap().clear(); }

  bool Contains(const_pointer t) const {
    size_type h = GetPositionOf(t);
    return (h != IntrusiveHeapLink::kNotMember) && (h < size()) &&
           heap()[h] == t;
  }

  void reserve(size_type n) { heap().reserve(n); }

  size_type capacity() const { return heap().capacity(); }

  allocator_type get_allocator() const { return rep_.heap_.get_allocator(); }

 private:
  typedef std::vector<pointer, allocator_type> heap_type;

  // Empty base class optimization for pointer_compare and link_access.
  // The heap_ data member retains a copy of the allocator, so it is not
  // stored explicitly.
  struct Rep : pointer_compare_type, link_access_type {
    explicit Rep(const pointer_compare_type& cmp,
                 const link_access_type& link_access,
                 const allocator_type& alloc)
        : pointer_compare_type(cmp),
          link_access_type(link_access),
          heap_(alloc) {}
    heap_type heap_;  // NOLINT
  };

  const pointer_compare_type& compare() const { return rep_; }

  const link_access_type& link_access() const { return rep_; }

  const heap_type& heap() const { return rep_.heap_; }
  heap_type& heap() { return rep_.heap_; }

  size_type GetPositionOf(const_pointer t) const {
    return link_access().Get(t).get();
  }

  void SetPositionOf(pointer t, size_type pos) const {
    return link_access().Set(t, IntrusiveHeapLink(pos));
  }

  void FixHeapUp(pointer t) {
    size_type h = GetPositionOf(t);
    while (h != 0) {
      size_type parent = (h - 1) >> 1;
      if (compare()(heap()[parent], t)) {
        break;
      }
      heap()[h] = heap()[parent];
      SetPositionOf(heap()[h], h);
      h = parent;
    }
    heap()[h] = t;
    SetPositionOf(t, h);
  }

  void FixHeapDown(pointer t) {
    size_type h = GetPositionOf(t);
    for (;;) {
      size_type kid = (h << 1) + 1;
      if (kid >= heap().size()) {
        break;
      }
      if (kid + 1 < heap().size() && compare()(heap()[kid + 1], heap()[kid])) {
        ++kid;
      }
      if (compare()(t, heap()[kid])) {
        break;
      }
      heap()[h] = heap()[kid];
      SetPositionOf(heap()[h], h);
      h = kid;
    }

    heap()[h] = t;
    SetPositionOf(t, h);
  }

  Rep rep_;
};

bool _IsTPUPartitionedInput(const Node* node) {
  return (node->type_string() == kTPUPartitionedInput) ||
         (node->type_string() == kTPUPartitionedInputV2);
}

bool _IsTPUPartitionedOutput(const Node* node) {
  return (node->type_string() == kTPUPartitionedOutput) ||
         (node->type_string() == kTPUPartitionedOutputV2);
}

std::string CoreDeviceLabel(int core) {
  return absl::StrCat("/device:", DEVICE_TPU_REPLICATED_CORE, ":", core);
}

// Creates a unique node name with a particular prefix.
std::string UniqueNodeName(absl::string_view prefix, Graph* graph) {
  return graph->NewName(absl::StrCat(prefix, "/_", internal::GetNodeId()));
}

absl::Status SetNodeDeviceForTPUCommunication(
    DeviceNameUtils::ParsedName device, const std::string& target_device_type,
    Node* node) {
  TF_RET_CHECK(device.has_type && device.type == DEVICE_TPU_NODE);
  TF_RET_CHECK(device.has_id);
  TF_RET_CHECK(HasNodeAttr(node->def(), kXlaHasHostTransferAttrName));

  // Store the device instance as an attr on the Node.
  TF_RETURN_IF_ERROR(SetDeviceOrdinalAttributeForNode(node, device.id));

  // Place the execute Op on the TPU_SYSTEM device so it can access the cache of
  // compiled protos in the resource manager.
  device.type = target_device_type;
  device.id = 0;

  node->set_assigned_device_name(DeviceNameUtils::ParsedNameToString(device));
  return absl::OkStatus();
}

// Iterate over the nodes in the original graph and find all the TPUReplicate
// nodes, and all the nodes that are part of outside_compilation clusters.
absl::Status FindTaggedNodes(
    Graph* graph, std::vector<Node*>* replicate_nodes,
    std::map<std::string, DistributedTPURewritePass::OutsideCompilationNodeMap>*
        outside_compilation_nodes,
    std::map<std::string, std::vector<Node*>>*
        head_tail_outside_compilation_nodes) {
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == "_TPUReplicate") {
      replicate_nodes->push_back(node);
      const AttrValue* cluster_attr = node->attrs().Find(kTPUReplicateAttr);
      if (cluster_attr == nullptr) {
        return absl::InternalError(absl::StrCat("TPUReplicate node ",
                                                node->name(), " has no ",
                                                kTPUReplicateAttr, " attr."));
      } else {
        const std::string& cluster = cluster_attr->s();
        if (cluster.empty()) {
          return absl::InternalError(absl::StrCat("Attr ", kTPUReplicateAttr,
                                                  " on node ", node->name(),
                                                  " has no string value."));
        }
        if (outside_compilation_nodes->find(cluster) !=
            outside_compilation_nodes->end()) {
          return absl::InternalError(absl::StrCat(
              "TPUReplicate node ", node->name(), " has ", kTPUReplicateAttr,
              " attr value '", cluster,
              "' which is a duplicate of another TPUReplicate node in the "
              "graph."));
        }
        (*outside_compilation_nodes)[cluster] =
            DistributedTPURewritePass::OutsideCompilationNodeMap();
        (*head_tail_outside_compilation_nodes)[cluster] = std::vector<Node*>();
      }
    }
  }
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() != "_TPUReplicate") {
      const AttrValue* cluster_attr = node->attrs().Find(kTPUReplicateAttr);
      const AttrValue* outside_compilation_attr =
          node->attrs().Find(kOutsideCompilationAttr);
      if (cluster_attr == nullptr) {
        if (outside_compilation_attr != nullptr) {
          return absl::InternalError(absl::StrCat(
              "Node ", node->name(), " has ", kOutsideCompilationAttr,
              " attr but no ", kTPUReplicateAttr, " attr."));
        }
      } else {
        const std::string& cluster = cluster_attr->s();
        if (cluster.empty()) {
          return absl::InternalError(absl::StrCat("Attr ", kTPUReplicateAttr,
                                                  " on node ", node->name(),
                                                  " has no string value."));
        }
        const auto iter = outside_compilation_nodes->find(cluster);
        if (iter == outside_compilation_nodes->end()) {
          return absl::InternalError(absl::StrCat(
              "Attr ", kTPUReplicateAttr, " on node ", node->name(),
              " does not correspond to a TPUReplicate node."));
        }
        if (outside_compilation_attr == nullptr) {
          return absl::InternalError(
              absl::StrCat("Node ", node->name(), " has ", kTPUReplicateAttr,
                           " attr but no ", kOutsideCompilationAttr, " attr."));
        }
        const std::string& oc_cluster = outside_compilation_attr->s();
        if (oc_cluster.empty()) {
          return absl::InternalError(
              absl::StrCat("Attr ", kOutsideCompilationAttr, " on node ",
                           node->name(), " has no string value."));
        }

        // Outside compilation cluster at head and tail of TPU computation has
        // already been moved to host and is already replicated. As so, do not
        // replicate outside compilation nodes with replica id attribute.
        int replica_id;
        if (TryGetNodeAttr(node->def(), kXlaReplicaIdAttrName, &replica_id)) {
          const AttrValue* head_attr =
              node->attrs().Find("_xla_only_arg_or_oc_input");
          const AttrValue* tail_attr =
              node->attrs().Find("_xla_only_ret_or_oc_output");
          if (((head_attr != nullptr) && (head_attr->b())) ||
              ((tail_attr != nullptr) && (tail_attr->b()))) {
            // This is safe as this has the same keys as
            // outside_compilation_nodes which we already know has this key.
            (*head_tail_outside_compilation_nodes)[cluster].push_back(node);
          }
          continue;
        }
        iter->second[oc_cluster].push_back(node);
      }
    }
  }
  return absl::OkStatus();
}

// Helper class to spread TPU computation arguments and return values
// across cores.
// If all shapes are fully defined, balance by their size.
// If some of them are not fully defined, the undefined shapes size will
// be estimated with the average size of the fully defined ones.
// If none are defined, fall back to round-robin.
class TensorDevicePlacer {
 public:
  // Creates a TensorDevicePlacer object to distribute arguments or
  // return values to a set of num_devices devices, where the types and
  // the inferred shapes of the inputs (arguments or return values) are
  // passed in types and shapes.
  TensorDevicePlacer(int64_t num_devices, const DataTypeVector& types,
                     const std::vector<InferredShape>& shapes)
      : index_nodes_(num_devices), sizes_(types.size()) {
    int64_t total_size = 0;
    int64_t num_defined = 0;
    for (int64_t i = 0; i < types.size(); ++i) {
      sizes_[i] = GetInferredShapeSize(shapes[i], types[i]);
      if (sizes_[i] >= 0) {
        total_size += sizes_[i];
        ++num_defined;
      }
    }
    // If a shape is undefined, select a size for it which is the average
    // of the defined shapes. If no shapes are defined, assign 1 so that we
    // get round-robin behavior.
    int64_t undefined_shape_size =
        (num_defined > 0) ? total_size / num_defined : 1;
    for (int64_t i = 0; i < sizes_.size(); ++i) {
      if (sizes_[i] < 0) {
        sizes_[i] = undefined_shape_size;
      }
    }

    for (int64_t i = 0; i < num_devices; ++i) {
      heap_.Push(&index_nodes_[i]);
    }
  }

  // Reports that the argument/return-value at index has been assigned
  // by the user to a given device.
  void ReportDeviceAssigned(int64_t device, int64_t index) {
    if (device >= index_nodes_.size()) {
      LOG(FATAL) << "Sharding assignment is out of bounds. "  // Crash OK
                    "Check that the number of nodes is properly set.";
    }
    DeviceNode* node = &index_nodes_.at(device);
    node->size += sizes_.at(index);
    heap_.Adjust(node);
  }

  // Retrieves the device at which the argument/return-value at index
  // should be assigned to.
  int64_t RetrieveAssignment(int64_t index) {
    DeviceNode* node = heap_.top();
    int64_t device = node - index_nodes_.data();
    node->size += sizes_.at(index);
    heap_.Adjust(node);
    return device;
  }

 private:
  struct DeviceNode {
    struct Compare {
      // Compare functor to implement a min heap using the ::gtl::IntrusiveHeap
      // infrastructure.
      bool operator()(const DeviceNode* lhs, const DeviceNode* rhs) const {
        return lhs->size < rhs->size;
      }
    };

    IntrusiveHeapLink heap;
    int64_t size = 0;
  };

  static int64_t GetInferredShapeSize(const InferredShape& ishape,
                                      DataType dtype) {
    return ishape.shape.IsFullyDefined()
               ? ishape.shape.num_elements() * DataTypeSize(dtype)
               : -1;
  }

  std::vector<DeviceNode> index_nodes_;
  IntrusiveHeap<DeviceNode, typename DeviceNode::Compare> heap_;
  std::vector<int64_t> sizes_;
};

absl::Status ValidateCoreNumber(int64_t core, int64_t num_cores_per_replica) {
  if (core < 0 || core >= num_cores_per_replica) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid core ID: ", core, ". The valid core IDs are [0..",
                     num_cores_per_replica, ")"));
  }
  return absl::OkStatus();
}

absl::Status FindHostComputeKeyPlaceholderNodes(
    const Graph* graph, const std::vector<Node*>& replicate_nodes,
    std::unordered_map<std::string, Node*>* host_compute_key_placeholder_map) {
  host_compute_key_placeholder_map->clear();
  for (const auto node : replicate_nodes) {
    (*host_compute_key_placeholder_map)[node->name()] = nullptr;
  }

  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == "Placeholder" &&
        absl::EndsWith(node->name(), "_key_placeholder")) {
      const AttrValue* call_node_attr =
          node->attrs().Find("_host_compute_call_node");
      if (call_node_attr != nullptr) {
        auto iter = host_compute_key_placeholder_map->find(call_node_attr->s());
        if (iter == host_compute_key_placeholder_map->end()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Node ", node->name(), " has _host_compute_call_node attribute '",
              call_node_attr->s(), "' that doesn't correspond to a call node"));
        }
        if (iter->second != nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("Key placeholder node ", iter->second->name(),
                           " for call node ", call_node_attr->s(),
                           " previously found as ", iter->second->name()));
        }
        iter->second = node;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ReplaceCompilationResultNodeWithIdentity(Graph* graph,
                                                      Node** node) {
  Node* old_node = *node;
  // We want to replace the node with an identity node with the same name.
  const std::string& node_name = old_node->name();

  // Create identity node.
  TF_ASSIGN_OR_RETURN(
      Node * id_node,
      BuildIdentityNode(graph, node_name, DT_STRING,
                        /*input=*/nullptr, /*requested_device=*/""));

  // No incoming edges are copied as a new one will be added from compile node
  // to id_node.

  // Copy outgoing edges to the id node.
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    Node* dst = edge->dst();
    int src_output = edge->src_output();
    int dst_input = edge->dst_input();

    if (src_output == Graph::kControlSlot) {
      graph->AddControlEdge(id_node, dst);
    } else {
      graph->AddEdge(id_node, src_output, dst, dst_input);
    }
    graph->RemoveEdge(edge);
  }
  graph->RemoveNode(old_node);

  *node = id_node;
  return absl::OkStatus();
}

absl::Status GetStepMarkerLocation(
    const Node& replicate_node,
    xla::DebugOptions::StepMarkerLocation* location) {
  std::string step_marker_location_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(), "step_marker_location",
                                 &step_marker_location_attr));
  if (step_marker_location_attr.empty()) {
    *location = xla::DebugOptions::STEP_MARK_AT_ENTRY;
  } else {
    if (!xla::DebugOptions::StepMarkerLocation_Parse(step_marker_location_attr,
                                                     location)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Malformed step_marker_location: ", step_marker_location_attr));
    }
  }
  return absl::OkStatus();
}

// Updates contents of the function with `function_name` in function library
// definition `flib_def` to `new_graph`. This is required when graph
// transformation happens inside a function call body.
absl::Status UpdateFunctionLibDefinition(const Graph& new_graph,
                                         const std::string& function_name,
                                         FunctionLibraryDefinition* flib_def) {
  FunctionDef graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(new_graph, function_name, &graph_fdef));
  TF_RETURN_IF_ERROR(flib_def->ReplaceFunction(function_name, graph_fdef));
  return absl::OkStatus();
}

struct NodeOut {
  Node* node;
  int index;
};

struct ShardedInputIndex {
  int replica_id;
  int argument_index;

  bool operator<(const ShardedInputIndex& rhs) const {
    return std::tie(replica_id, argument_index) <
           std::tie(rhs.replica_id, rhs.argument_index);
  }
};

struct ShardedPerHostInputIndex {
  std::string host_device;
  int argument_index;
  bool operator<(const ShardedPerHostInputIndex& rhs) const {
    return std::tie(host_device, argument_index) <
           std::tie(rhs.host_device, rhs.argument_index);
  }
  bool operator==(const ShardedPerHostInputIndex& rhs) const {
    return (argument_index == rhs.argument_index) &&
           (host_device == rhs.host_device);
  }
};

struct ShardedInputInfo {
  // Split node that would be connected to tiled input Node.
  Node* split_node;
  // List of splits nodes and output index of the split node from which sharded
  // input will be connected to the TPUExecute node. The inputs are ordered by
  // logical core ids.
  std::vector<NodeOut> sharded_inputs;
};

// Adds pad node after split node to graph for uneven sharding tiled inputs.
// |graph| owns the returned Node* instance.
absl::StatusOr<Node*> CreatePadNode(const int padding, const int num_dims,
                                    const int split_dim, DataType dtype,
                                    Node* control_predecessor, Node* split_node,
                                    const int split_index, Graph* graph) {
  // Add paddings node.
  absl::Status s;
  NodeDef paddings_def;
  paddings_def.set_name(
      graph->NewName(absl::StrCat(split_node->name(), "/paddings")));
  paddings_def.set_op("Const");
  AddNodeAttr("dtype", DT_INT32, &paddings_def);
  paddings_def.set_device(split_node->assigned_device_name());
  TensorProto sizes_tensor_proto;
  sizes_tensor_proto.set_dtype(DT_INT32);
  for (int i = 0; i < num_dims; ++i) {
    sizes_tensor_proto.add_int_val(0);
    if (i == split_dim) {
      sizes_tensor_proto.add_int_val(padding);
    } else {
      sizes_tensor_proto.add_int_val(0);
    }
  }
  TensorShape sizes_shape({num_dims, 2});
  sizes_shape.AsProto(sizes_tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", sizes_tensor_proto, &paddings_def);
  TF_ASSIGN_OR_RETURN(Node * paddings_node, graph->AddNode(paddings_def));

  // Add Pad node.
  NodeDef pad_def;
  pad_def.set_name(graph->NewName(
      absl::StrCat(split_node->name(), "/pad_shard_", split_index)));
  pad_def.set_op("Pad");
  pad_def.set_device(split_node->assigned_device_name());
  AddNodeAttr("T", dtype, &pad_def);
  AddNodeAttr("Tpaddings", DT_INT32, &pad_def);
  pad_def.add_input(absl::StrCat(split_node->name(), ":", split_index));
  pad_def.add_input(absl::StrCat(paddings_node->name(), ":0"));
  TF_ASSIGN_OR_RETURN(Node * pad_node, graph->AddNode(pad_def));
  pad_node->set_assigned_device_name(split_node->assigned_device_name());
  // Add edges for pad node.
  graph->AddEdge(split_node, split_index, pad_node, 0);
  graph->AddEdge(paddings_node, 0, pad_node, 1);
  graph->AddControlEdge(control_predecessor, pad_node);
  return pad_node;
}

// Adds split node and split dimension node to graph for sharding tiled inputs.
// |graph| owns the returned Node* instance.
absl::StatusOr<Node*> CreateSplitNode(const int num_splits, const int dim,
                                      const int num_dims, const int64_t padding,
                                      const int orig_src_output, DataType dtype,
                                      absl::string_view name_prefix,
                                      Node* control_predecessor, Node* orig_src,
                                      Graph* graph) {
  const std::string input_assigned_device = orig_src->assigned_device_name();
  Node* to_split_node = orig_src;
  int to_split_index = orig_src_output;
  if (padding > 0) {
    TF_ASSIGN_OR_RETURN(
        Node * pad_node,
        CreatePadNode(padding, num_dims, dim, dtype, control_predecessor,
                      orig_src, orig_src_output, graph));
    to_split_node = pad_node;
    to_split_index = 0;
  }

  // Add a split dimension node.
  NodeDef split_dim_def;
  split_dim_def.set_name(
      graph->NewName(absl::StrCat(name_prefix, "/split_dim")));
  split_dim_def.set_op("Const");
  split_dim_def.set_device(input_assigned_device);
  AddNodeAttr("dtype", DT_INT32, &split_dim_def);
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.add_int_val(dim);
  TensorShape shape({});
  shape.AsProto(tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensor_proto, &split_dim_def);
  TF_ASSIGN_OR_RETURN(Node * split_dim_node, graph->AddNode(split_dim_def));
  // Add a split node.
  NodeDef split_def;
  split_def.set_name(graph->NewName(absl::StrCat(name_prefix, "/split")));
  split_def.set_op("Split");
  split_def.set_device(input_assigned_device);
  AddNodeAttr("num_split", num_splits, &split_def);
  AddNodeAttr("T", dtype, &split_def);
  split_def.add_input(absl::StrCat(split_dim_node->name(), ":0"));
  split_def.add_input(absl::StrCat(to_split_node->name(), ":", to_split_index));
  TF_ASSIGN_OR_RETURN(Node * split_node, graph->AddNode(split_def));

  split_node->set_assigned_device_name(input_assigned_device);

  // If colocate the newly created split op to source node of input to TPU
  // computation.
  split_node->AddAttr(kColocationAttrName,
                      std::vector<std::string>{absl::StrCat(
                          kColocationGroupPrefix, orig_src->name())});

  graph->AddEdge(split_dim_node, 0, split_node, 0);
  graph->AddEdge(to_split_node, to_split_index, split_node, 1);

  // Add a control dependency from `control_predecessor` to newly created
  // constant node. This ensures that newly added split/split dim
  // nodes are placed inside correct while loop frames when TPUExecute
  // node is inside a host training loop.
  graph->AddControlEdge(control_predecessor, split_dim_node);
  return split_node;
}

int64_t GetPadding(const int split_dim, const int num_splits,
                   const PartialTensorShape& partial_tensor_shape) {
  // If dim dimension is not defined, no uneven sharding support.
  if (partial_tensor_shape.dim_size(split_dim) <= 0) {
    return 0;
  }
  int64_t per_split_size = tensorflow::MathUtil::CeilOfRatio<int64_t>(
      partial_tensor_shape.dim_size(split_dim), num_splits);
  int64_t total_padding =
      per_split_size * num_splits - partial_tensor_shape.dim_size(split_dim);
  return total_padding;
}

// Creates a set of splits nodes that shards tiled input node in graph.
absl::StatusOr<ShardedInputInfo> CreateOrGetSplitNodesForInputSharding(
    const xla::OpSharding& sharding, int orig_arg_num, DataType dtype,
    const PartialTensorShape& partial_tensor_shape, int replica_id,
    int orig_src_output, Node* orig_src, Node* control_predecessor,
    Graph* graph,
    std::map<ShardedInputIndex, ShardedInputInfo>*
        arg_index_to_sharded_input_map) {
  ShardedInputIndex input_index{replica_id, orig_arg_num};
  auto iter = arg_index_to_sharded_input_map->find(input_index);
  if (iter != arg_index_to_sharded_input_map->end()) {
    return iter->second;
  }
  // Maps input dimension and number of splits with which the
  // dimension sharded.
  TF_ASSIGN_OR_RETURN(auto split_dimension_map,
                      GetDimensionIndicesAndNumSplitsFromSharding(sharding));
  TF_RET_CHECK(!split_dimension_map.empty())
      << "Unnecessary sharding attribute found.";

  // For v1 while loop, nodes inside the loop body must either
  //  1) Have data edges from while loop input node.
  //  or
  //  2) Have direct control dependency from while loop input control
  //     node.
  //
  // As so, if we are adding Split node inside, while loop body,
  // we must manually add a control dependency to a node inside
  // a while loop (i.e. `control_predecessor`) to constant nodes
  // without data in-edges to make sure that added split nodes
  // have correct frame name. Else, placer will complain when
  // `BuildControlFlow()` is invoked.

  auto sharding_it = split_dimension_map.begin();
  std::queue<Node*> split_nodes_for_dimension;
  absl::flat_hash_map<Node*, int> node_to_split_dim;
  int split_dimension = sharding_it->first;
  int num_split = sharding_it->second;

  // Creates a tree of split nodes for sharding tiled inputs. Splits nodes
  // are created such that input data is sharded in row major order.
  // Split nodes at ith depth from the original input node represent nodes
  // that split the input data at ith dimension.
  TF_ASSIGN_OR_RETURN(
      Node * root_split_node,
      CreateSplitNode(
          num_split, split_dimension, partial_tensor_shape.dims(),
          GetPadding(split_dimension, num_split, partial_tensor_shape),
          orig_src_output, dtype,
          absl::StrCat("sharded_input/replica_", replica_id, "_dim_",
                       split_dimension),
          control_predecessor, orig_src, graph));
  sharding_it++;

  split_nodes_for_dimension.emplace(root_split_node);
  node_to_split_dim[root_split_node] = split_dimension;

  while (sharding_it != split_dimension_map.end()) {
    split_dimension = sharding_it->first;
    num_split = sharding_it->second;
    int num_split_nodes_in_dimension = split_nodes_for_dimension.size();
    for (int i = 0; i < num_split_nodes_in_dimension; ++i) {
      Node* input_split_node = split_nodes_for_dimension.front();
      split_nodes_for_dimension.pop();
      for (int src_output_index = 0;
           src_output_index < input_split_node->num_outputs();
           ++src_output_index) {
        TF_ASSIGN_OR_RETURN(
            Node * split_node,
            CreateSplitNode(
                num_split, split_dimension, partial_tensor_shape.dims(),
                GetPadding(split_dimension, num_split, partial_tensor_shape),
                src_output_index, dtype,
                absl::StrCat("sharded_input/replica_", replica_id, "_dim_",
                             split_dimension),
                control_predecessor, input_split_node, graph));
        split_nodes_for_dimension.emplace(split_node);
        node_to_split_dim[split_node] = split_dimension;
      }
    }
    sharding_it++;
  }

  // `split_nodes_for_dimension` now includes final split nodes
  // from which sharded data will be fed into TPUExcute nodes -- sorted by
  // row major order.
  std::vector<NodeOut> sharded_inputs_list(
      sharding.tile_assignment_devices_size());
  int64_t next_core_tile_index = 0;
  while (!split_nodes_for_dimension.empty()) {
    Node* split_node = split_nodes_for_dimension.front();
    split_nodes_for_dimension.pop();
    int num_splits;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(split_node->def(), "num_split", &num_splits));
    for (int out_index = 0; out_index < num_splits; ++out_index) {
      int64_t repeat_count =
          sharding.replicate_on_last_tile_dim()
              ? *sharding.tile_assignment_dimensions().rbegin()
              : 1;
      for (int64_t i = 0; i < repeat_count; ++i) {
        int64_t next_core =
            sharding.tile_assignment_devices(next_core_tile_index++);
        sharded_inputs_list[next_core] = NodeOut{split_node, out_index};
      }
    }
  }

  ShardedInputInfo sharded_input_info{root_split_node,
                                      std::move(sharded_inputs_list)};
  (*arg_index_to_sharded_input_map)[input_index] = sharded_input_info;
  return sharded_input_info;
}

// Creates a xla split node to shard an input, and adds that new node to a
// Graph.
absl::StatusOr<Node*> CreateXlaSplitOp(
    absl::string_view node_name, const bool is_resource, const NodeOut& input,
    const PartialTensorShape& partial_tensor_shape,
    const std::vector<Node*>& control_inputs,
    const std::vector<Node*>& control_outputs, const DataType dtype,
    const int num_shards, const xla::OpSharding& sharding, Graph* graph) {
  const std::string& input_assigned_device = input.node->assigned_device_name();
  NodeDef xla_split_def;
  xla_split_def.set_name(graph->NewName(node_name));
  xla_split_def.set_op(is_resource ? "ReadVariableXlaSplitND" : "XlaSplitND");
  xla_split_def.set_device(input_assigned_device);
  AddNodeAttr("T", dtype, &xla_split_def);
  AddNodeAttr("N", num_shards, &xla_split_def);
  const std::vector<int64_t> num_splits(
      sharding.tile_assignment_dimensions().begin(),
      sharding.replicate_on_last_tile_dim()
          ? std::prev(sharding.tile_assignment_dimensions().end())
          : sharding.tile_assignment_dimensions().end());
  AddNodeAttr("num_splits", num_splits, &xla_split_def);
  const int rank = sharding.replicate_on_last_tile_dim()
                       ? sharding.tile_assignment_dimensions_size() - 1
                       : sharding.tile_assignment_dimensions_size();
  std::vector<int32_t> paddings;
  paddings.reserve(rank);
  for (int dim = 0; dim < rank; ++dim) {
    paddings.push_back(GetPadding(dim, sharding.tile_assignment_dimensions(dim),
                                  partial_tensor_shape));
  }
  AddNodeAttr("paddings", paddings, &xla_split_def);

  if (!is_resource) {
    AddNodeAttr("_tpu_avoid_constant_fold", "not_used", &xla_split_def);
    AddNodeAttr(kColocationAttrName,
                std::vector<std::string>{
                    absl::StrCat(kColocationGroupPrefix, input.node->name())},
                &xla_split_def);
  }

  TF_ASSIGN_OR_RETURN(Node * xla_split, graph->AddNode(xla_split_def));
  if (is_resource) {
    xla_split->set_requested_device(input.node->requested_device());
  }
  xla_split->set_assigned_device_name(input_assigned_device);
  graph->AddEdge(input.node, input.index, xla_split, 0);
  for (Node* control_input : control_inputs) {
    graph->AddControlEdge(control_input, xla_split);
  }
  for (Node* control_output : control_outputs) {
    graph->AddControlEdge(xla_split, control_output);
  }
  return xla_split;
}

// Creates a sharded tensor list for all input shards of an input with sharding.
absl::StatusOr<std::vector<NodeOut>> ShardInputWithXlaSplitOp(
    absl::string_view node_name, const bool is_resource, const NodeOut& input,
    const PartialTensorShape& partial_tensor_shape,
    const std::vector<Node*>& control_inputs,
    const std::vector<Node*>& control_outputs, const DataType dtype,
    const xla::OpSharding& sharding, Graph* graph) {
  const int repeat = sharding.replicate_on_last_tile_dim()
                         ? *sharding.tile_assignment_dimensions().rbegin()
                         : 1;
  const int num_shards = sharding.tile_assignment_devices_size() / repeat;

  TF_ASSIGN_OR_RETURN(
      Node * xla_split,
      CreateXlaSplitOp(node_name, is_resource, input, partial_tensor_shape,
                       control_inputs, control_outputs, dtype, num_shards,
                       sharding, graph));

  std::vector<NodeOut> sharded_inputs_list(
      sharding.tile_assignment_devices_size());

  for (int i = 0; i < num_shards; ++i) {
    for (int j = 0; j < repeat; ++j) {
      const int index = i * repeat + j;
      const int core = sharding.tile_assignment_devices(index);
      sharded_inputs_list[core] = {xla_split, i};
    }
  }

  return sharded_inputs_list;
}

// Creates an XlaSplitND op to shard a per-replica arg.
absl::StatusOr<ShardedInputInfo> CreateOrGetXlaSplitNodeForShardedPerReplicaArg(
    const xla::OpSharding& sharding, const int replica_id,
    const int orig_arg_num, DataType dtype,
    const PartialTensorShape& partial_tensor_shape, Node* orig_src,
    const int orig_src_output, Graph* graph,
    std::map<ShardedInputIndex, ShardedInputInfo>*
        arg_index_to_sharded_input_map) {
  ShardedInputIndex input_index{replica_id, orig_arg_num};
  auto iter = arg_index_to_sharded_input_map->find(input_index);
  if (iter != arg_index_to_sharded_input_map->end()) {
    return iter->second;
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<NodeOut> sharded_inputs_list,
      ShardInputWithXlaSplitOp(
          absl::StrCat(orig_src->name(), "/replica_", replica_id, "_split"),
          /*is_resource=*/false, /*input=*/{orig_src, orig_src_output},
          partial_tensor_shape, /*control_inputs=*/{}, /*control_outputs=*/{},
          dtype, sharding, graph));

  ShardedInputInfo sharded_input_info{nullptr, std::move(sharded_inputs_list)};
  (*arg_index_to_sharded_input_map)[input_index] = sharded_input_info;
  return sharded_input_info;
}

// Creates an XlaSplitND op to shard a distributed arg.
absl::StatusOr<ShardedInputInfo> CreateOrGetXlaSplitNodeForDistributedArg(
    const xla::OpSharding& sharding, const int num_replicas,
    const int replica_id, const int orig_arg_num, DataType dtype,
    const PartialTensorShape& partial_tensor_shape, Node* orig_src,
    const int orig_src_output, Graph* graph,
    std::map<ShardedInputIndex, ShardedInputInfo>*
        arg_index_to_sharded_input_map) {
  ShardedInputIndex input_index{replica_id, orig_arg_num};
  auto iter = arg_index_to_sharded_input_map->find(input_index);
  if (iter != arg_index_to_sharded_input_map->end()) {
    return iter->second;
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<NodeOut> sharded_inputs_list,
      ShardInputWithXlaSplitOp(
          absl::StrCat(orig_src->name(), "/distributed_split"),
          /*is_resource=*/false, /*input=*/{orig_src, orig_src_output},
          partial_tensor_shape, /*control_inputs=*/{}, /*control_outputs=*/{},
          dtype, sharding, graph));

  ShardedInputInfo sharded_input_info{nullptr, std::move(sharded_inputs_list)};
  for (int replica = 0; replica < num_replicas; ++replica) {
    (*arg_index_to_sharded_input_map)[{replica, orig_arg_num}] =
        sharded_input_info;
  }
  return sharded_input_info;
}

// Creates an ReadVariableXlaSplitND op to shard a variable arg.
absl::StatusOr<ShardedInputInfo> CreateOrGetXlaSplitNodeForVariableArg(
    const xla::OpSharding& sharding, const int num_replicas,
    const int replica_id, const int orig_arg_num, DataType dtype,
    const PartialTensorShape& partial_tensor_shape, Node* orig_src,
    const int orig_src_output, Graph* graph,
    std::vector<Node*>* to_be_removed_nodes,
    std::map<ShardedInputIndex, ShardedInputInfo>*
        arg_index_to_sharded_input_map) {
  ShardedInputIndex input_index{replica_id, orig_arg_num};
  auto iter = arg_index_to_sharded_input_map->find(input_index);
  if (iter != arg_index_to_sharded_input_map->end()) {
    return iter->second;
  }

  DCHECK_EQ(orig_src->type_string(), "ReadVariableOp");
  std::vector<Node*> control_outputs;
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* edge : orig_src->out_edges()) {
    if (edge->IsControlEdge()) {
      control_outputs.push_back(edge->dst());
    }
    edges_to_remove.push_back(edge);
  }

  to_be_removed_nodes->push_back(orig_src);

  const Edge* resource = nullptr;
  TF_RETURN_IF_ERROR(orig_src->input_edge(0, &resource));

  std::vector<Node*> control_inputs;
  for (const Edge* edge : orig_src->in_edges()) {
    if (edge->IsControlEdge()) {
      control_inputs.push_back(edge->src());
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<NodeOut> sharded_inputs_list,
      ShardInputWithXlaSplitOp(
          absl::StrCat(resource->src()->name(), "/read_variable_split"),
          /*is_resource=*/true,
          /*input=*/{resource->src(), resource->src_output()},
          partial_tensor_shape, control_inputs, control_outputs, dtype,
          sharding, graph));

  for (const Edge* edge : edges_to_remove) {
    graph->RemoveControlEdge(edge);
  }

  DCHECK(orig_src->out_edges().empty());

  ShardedInputInfo sharded_input_info{nullptr, std::move(sharded_inputs_list)};
  for (int replica = 0; replica < num_replicas; ++replica) {
    ShardedInputIndex idx{replica, orig_arg_num};
    // Refrain from overwriting, if dummy inputs were already placed instead.
    arg_index_to_sharded_input_map->insert({idx, sharded_input_info});
  }
  return sharded_input_info;
}

// Creates a concat node to be used for aggregating sharded retvals across
// logical cores.
absl::StatusOr<Node*> CreateConcatNode(int dim, int num_splits, DataType dtype,
                                       absl::string_view name_prefix,
                                       const std::vector<NodeOut>& inputs,
                                       Graph* graph, absl::string_view device) {
  // Add a Concat dim node.
  NodeDef concat_dim_def;
  concat_dim_def.set_name(
      graph->NewName(absl::StrCat(name_prefix, "/concat_dim")));
  concat_dim_def.set_op("Const");
  AddNodeAttr("dtype", DT_INT32, &concat_dim_def);
  concat_dim_def.set_device(std::string(device));
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.add_int_val(dim);
  TensorShape shape({});
  shape.AsProto(tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensor_proto, &concat_dim_def);
  TF_ASSIGN_OR_RETURN(Node * concat_dim_node, graph->AddNode(concat_dim_def));

  // Add a Concat node.
  NodeDef concat_def;
  concat_def.set_name(graph->NewName(absl::StrCat(name_prefix, "/concat")));
  concat_def.set_op("Concat");
  AddNodeAttr("N", num_splits, &concat_def);
  AddNodeAttr("T", dtype, &concat_def);
  concat_def.add_input(absl::StrCat(concat_dim_node->name(), ":0"));
  concat_def.set_device(std::string(device));
  for (const auto& i : inputs) {
    concat_def.add_input(absl::StrCat(i.node->name(), ":", i.index));
  }
  TF_ASSIGN_OR_RETURN(Node * concat_node, graph->AddNode(concat_def));

  graph->AddEdge(concat_dim_node, 0, concat_node, 0);

  // 0th input to concat node is a concat dim node. So we start from 1st input
  // and add all input edges.
  int dst_input = 1;
  for (const auto& i : inputs) {
    graph->AddEdge(i.node, i.index, concat_node, dst_input);
    ++dst_input;
  }
  return concat_node;
}

// Adds slice node after concat node to graph for uneven sharding tiled inputs.
absl::StatusOr<Node*> CreateSliceNode(DataType dtype,
                                      const PartialTensorShape& shape,
                                      Node* concat_node,
                                      const int concat_out_index, Graph* graph,
                                      absl::string_view device) {
  absl::Status s;
  // Add begin node for concat.
  NodeDef begin_def;
  begin_def.set_name(
      graph->NewName(absl::StrCat(concat_node->name(), "/slice_begin")));
  begin_def.set_op("Const");
  AddNodeAttr("dtype", DT_INT32, &begin_def);
  begin_def.set_device(std::string(device));
  TensorProto begin_tensor_proto;
  begin_tensor_proto.set_dtype(DT_INT32);
  for (int i = 0; i < shape.dims(); ++i) {
    begin_tensor_proto.add_int_val(0);
  }
  TensorShape begin_shape({shape.dims()});
  begin_shape.AsProto(begin_tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", begin_tensor_proto, &begin_def);
  TF_ASSIGN_OR_RETURN(Node * begin_node, graph->AddNode(begin_def));

  // Add size node.
  NodeDef size_def;
  size_def.set_name(
      graph->NewName(absl::StrCat(concat_node->name(), "/slice_size")));
  size_def.set_op("Const");
  AddNodeAttr("dtype", DT_INT32, &size_def);
  size_def.set_device(std::string(device));
  TensorProto sizes_tensor_proto;
  sizes_tensor_proto.set_dtype(DT_INT32);
  for (int i = 0; i < shape.dims(); ++i) {
    sizes_tensor_proto.add_int_val(shape.dim_size(i));
  }
  TensorShape sizes_shape({shape.dims()});
  sizes_shape.AsProto(sizes_tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", sizes_tensor_proto, &size_def);
  TF_ASSIGN_OR_RETURN(Node * size_node, graph->AddNode(size_def));

  // Add Slice node.
  NodeDef slice_def;
  slice_def.set_name(
      graph->NewName(absl::StrCat(concat_node->name(), "/slice")));
  slice_def.set_op("Slice");
  slice_def.set_device(std::string(device));
  AddNodeAttr("T", dtype, &slice_def);
  AddNodeAttr("Index", DT_INT32, &slice_def);
  slice_def.add_input(absl::StrCat(concat_node->name(), ":", concat_out_index));
  slice_def.add_input(absl::StrCat(begin_node->name(), ":0"));
  slice_def.add_input(absl::StrCat(size_node->name(), ":0"));
  TF_ASSIGN_OR_RETURN(Node * slice_node, graph->AddNode(slice_def));
  // Add edges for slice node.
  graph->AddEdge(concat_node, concat_out_index, slice_node, 0);
  graph->AddEdge(begin_node, 0, slice_node, 1);
  graph->AddEdge(size_node, 0, slice_node, 2);
  return slice_node;
}

// Creates a set of Concat nodes that aggregates sharded outputs from TPUExecute
// nodes into a single output. Sharded outputs are concatenated along row major
// order. That is, tiled output along 0th dimension will be concatenated last.
absl::StatusOr<Node*> CreateConcatNodesForRetval(
    const xla::OpSharding& sharding, DataType dtype,
    const PartialTensorShape& inferred_shape, int replica_id,
    const std::vector<NodeOut>& orig_inputs, Graph* graph,
    absl::string_view device) {
  TF_ASSIGN_OR_RETURN(auto split_dimension_map,
                      GetDimensionIndicesAndNumSplitsFromSharding(sharding));
  std::vector<NodeOut> inputs_to_sharded_retval = orig_inputs;
  bool has_paddings = false;

  for (auto it = split_dimension_map.rbegin(); it != split_dimension_map.rend();
       it++) {
    auto dim = it->first;
    auto num_splits = it->second;

    int num_concat_nodes = inputs_to_sharded_retval.size() / num_splits;
    int input_index_to_concat_node = 0;

    std::vector<NodeOut> new_concat_nodes;
    for (int i = 0; i < num_concat_nodes; ++i) {
      auto concat_input_it =
          inputs_to_sharded_retval.begin() + input_index_to_concat_node;
      std::vector<NodeOut> inputs(concat_input_it,
                                  concat_input_it + num_splits);
      input_index_to_concat_node += num_splits;

      TF_ASSIGN_OR_RETURN(
          Node * concat_node,
          CreateConcatNode(
              dim, num_splits, dtype,
              absl::StrCat("sharded_output/replica_", replica_id, "_dim_", dim),
              inputs, graph, device));
      int64_t paddings = GetPadding(dim, num_splits, inferred_shape);
      has_paddings |= paddings > 0;
      new_concat_nodes.emplace_back(NodeOut{concat_node, 0});
    }
    inputs_to_sharded_retval = new_concat_nodes;
  }

  TF_RET_CHECK(inputs_to_sharded_retval.size() == 1);
  if (has_paddings) {
    TF_ASSIGN_OR_RETURN(Node * slice_node,
                        CreateSliceNode(dtype, inferred_shape,
                                        inputs_to_sharded_retval.at(0).node,
                                        /*concat_out_index*/ 0, graph, device));
    return slice_node;
  }
  return inputs_to_sharded_retval.at(0).node;
}

absl::StatusOr<Node*> CreateXlaConcatNode(
    const xla::OpSharding& sharding, const int replica_id, DataType dtype,
    const PartialTensorShape& partial_tensor_shape,
    const std::vector<NodeOut>& orig_inputs, absl::string_view device,
    Graph* graph) {
  NodeDef xla_concat_def;
  xla_concat_def.set_name(graph->NewName(
      absl::StrCat("sharded_output/replica_", replica_id, "_concat")));
  xla_concat_def.set_op("XlaConcatND");
  xla_concat_def.set_device(std::string(device));
  AddNodeAttr("T", dtype, &xla_concat_def);
  AddNodeAttr("N", static_cast<int64_t>(orig_inputs.size()), &xla_concat_def);
  const std::vector<int64_t> num_concats(
      sharding.tile_assignment_dimensions().begin(),
      sharding.replicate_on_last_tile_dim()
          ? std::prev(sharding.tile_assignment_dimensions().end())
          : sharding.tile_assignment_dimensions().end());
  AddNodeAttr("num_concats", num_concats, &xla_concat_def);
  const int rank = sharding.replicate_on_last_tile_dim()
                       ? sharding.tile_assignment_dimensions_size() - 1
                       : sharding.tile_assignment_dimensions_size();
  std::vector<int32_t> paddings;
  paddings.reserve(rank);
  for (int dim = 0; dim < rank; ++dim) {
    paddings.push_back(GetPadding(dim, sharding.tile_assignment_dimensions(dim),
                                  partial_tensor_shape));
  }
  AddNodeAttr("paddings", paddings, &xla_concat_def);

  TF_ASSIGN_OR_RETURN(Node * xla_concat, graph->AddNode(xla_concat_def));
  for (int i = 0, e = orig_inputs.size(); i < e; ++i) {
    const NodeOut& input = orig_inputs[i];
    graph->AddEdge(input.node, input.index, xla_concat, i);
  }
  return xla_concat;
}

// Set the padding ops the same devices as the original inputs. If the original
// inputs are on TPUs, the padding ops will be placed on TPUs and XLA on demand
// mode will be triggered, so we don't need to copy the data back to the host
// to do the padding.
absl::Status SetPaddingNodesDevices(Graph* graph) {
  for (Node* n : graph->op_nodes()) {
    bool tpu_padding_attr;
    if (n->type_string() == "Pad" &&
        GetNodeAttr(n->attrs(), kPostDeviceRewriteAttr, &tpu_padding_attr)
            .ok()) {
      Node* unpadded_input;
      TF_RETURN_IF_ERROR(n->input_node(0, &unpadded_input));

      const std::string& requested_device = unpadded_input->requested_device();
      const std::string& assigned_device =
          unpadded_input->assigned_device_name();
      if (!requested_device.empty() || !assigned_device.empty()) {
        // The output nodes of the original unpadded inputs include the padded
        // inputs and real shapes of inputs, we assign those to the same device
        // as the original inputs.
        for (Node* out : unpadded_input->out_nodes()) {
          if (GetNodeAttr(out->attrs(), kPostDeviceRewriteAttr,
                          &tpu_padding_attr)
                  .ok()) {
            out->set_requested_device(requested_device);
            out->set_assigned_device_name(assigned_device);
          }
        }
        // There might be a tf.shape node added before TPUCompileOp, we need to
        // set its device as well.
        for (Node* out : n->out_nodes()) {
          if (n->type_string() == "Shape") {
            out->set_requested_device(requested_device);
            out->set_assigned_device_name(assigned_device);
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

bool IsTpuDevice(absl::string_view device_string) {
  DeviceNameUtils::ParsedName device;
  return DeviceNameUtils::ParseFullName(device_string, &device) &&
         device.type == DEVICE_TPU_NODE;
}

bool CanAcceptTPUDevicePropagation(const Node& node) {
  // A set of device ops can be placed on TPU. There is no strict rule of
  // thumb to decide which ops should be in the list, but empirically they are
  // mostly dummy ops like Identity-like ops or control flow related ops.
  // However one can add also add other ops like Pad to allow data stay on TPU.
  static const auto place_on_tpu_ops = new absl::flat_hash_set<std::string>(
      {"Identity", "IdentityN", "Enter", "Exit", "Switch", "Merge",
       "NextIteration", "Shape", "_Retval"});
  return place_on_tpu_ops->contains(node.type_string());
}

xla::OpMetadata CreateOpMetadataFromNode(const Node& node) {
  xla::OpMetadata metadata;
  metadata.set_op_type(node.type_string());
  metadata.set_op_name(node.name());
  return metadata;
}

// Helper struct holding node (nullable) and associated sharding.
struct NodeAndSharding {
  explicit NodeAndSharding(const Node* node, const xla::OpSharding& sharding)
      : node(node), sharding(sharding) {}

  const Node* node;
  xla::OpSharding sharding;
};

// Validate sharding configuration derived from XlaSharding attribute.
// Infer the core id from the OpSharding, if necessary.
absl::Status ParseAndValidateSharding(const NodeAndSharding& node_and_sharding,
                                      const int num_cores_per_replica,
                                      int64_t* inferred_core_id,
                                      std::optional<NodeAndSharding>* result) {
  if (node_and_sharding.sharding.type() == xla::OpSharding::MAXIMAL) {
    int64_t core_annotation =
        node_and_sharding.sharding.tile_assignment_devices(0);
    TF_RETURN_IF_ERROR(
        ValidateCoreNumber(core_annotation, num_cores_per_replica));
    if (*inferred_core_id == -1 || *inferred_core_id > core_annotation) {
      *inferred_core_id = core_annotation;
      result->emplace(node_and_sharding);
    }
  } else {
    if (node_and_sharding.sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t core :
           node_and_sharding.sharding.tile_assignment_devices()) {
        TF_RETURN_IF_ERROR(ValidateCoreNumber(core, num_cores_per_replica));
      }
    }

    if (!result->has_value()) {
      *result = node_and_sharding;
    } else {
      std::string result_value_serialized;
      xla::OpSharding result_value = result->value().sharding;
      result_value.clear_metadata();
      SerializeToStringDeterministic(result_value, &result_value_serialized);

      std::string sharding_serialized;
      xla::OpSharding sharding = node_and_sharding.sharding;
      sharding.clear_metadata();
      SerializeToStringDeterministic(sharding, &sharding_serialized);

      // TODO(lyandy): Choose the more granular sharding instead of always
      // assigning to core 0 (maximal).
      if (result_value_serialized != sharding_serialized) {
        // We see different shardings, assign to core 0.
        auto core_zero_sharding = xla::sharding_builder::AssignDevice(0);
        DCHECK_NE(node_and_sharding.node, nullptr);
        *core_zero_sharding.add_metadata() =
            CreateOpMetadataFromNode(*node_and_sharding.node);
        result->emplace(
            NodeAndSharding(node_and_sharding.node, core_zero_sharding));
      }
    }
  }
  return absl::OkStatus();
}

// As XlaSharding node may be followed by Cast op or an Identity op,
// recursively walk the graph and aggregate nodes connectd to
// |input_node| or Cast/Identity op following the |input_node|.
void FindNodesMaybeContainingShardingInfo(const Node& input_node,
                                          std::vector<const Node*>* nodes) {
  if (input_node.IsIdentity() || input_node.type_string() == "Cast") {
    for (const Node* connected_node : input_node.out_nodes())
      FindNodesMaybeContainingShardingInfo(*connected_node, nodes);
  }
  nodes->emplace_back(&input_node);
}

// Parse sharding configuration from |node| or it's adjacent nodes.
// XlaSharding configuration may be derived from
//   a) Connected Identity op node.
//   b) Connected Cast op node.
absl::StatusOr<std::optional<NodeAndSharding>>
ParseInputShardingFromAdjacentNode(const int num_cores_per_replica,
                                   const Node& node) {
  // If |node| has `device` attribute or is a XlaSharding op,
  // return the parsed OpSharding.
  TF_ASSIGN_OR_RETURN(std::optional<xla::OpSharding> sharding,
                      ParseShardingFromDevice(node, num_cores_per_replica,
                                              /*add_metadata=*/true));
  if (sharding.has_value()) {
    return std::optional<NodeAndSharding>(NodeAndSharding(&node, *sharding));
  }

  // XlaShardingOp may be followed by an identity or followed by identity
  // and a Cast op.
  std::vector<const Node*> potential_nodes_with_input_sharding;
  FindNodesMaybeContainingShardingInfo(node,
                                       &potential_nodes_with_input_sharding);
  for (const Node* maybe_node_with_sharding_info :
       potential_nodes_with_input_sharding) {
    if (maybe_node_with_sharding_info->type_string() != "XlaSharding") continue;

    TF_ASSIGN_OR_RETURN(
        std::optional<xla::OpSharding> sharding_config,
        ParseShardingFromDevice(*maybe_node_with_sharding_info,
                                num_cores_per_replica, /*add_metadata=*/true));
    if (sharding_config.has_value()) {
      return std::optional<NodeAndSharding>(
          NodeAndSharding(maybe_node_with_sharding_info, *sharding_config));
    }
  }
  return std::optional<NodeAndSharding>();
}

// Walk the graph from an argument node to find OpSharding configuration
// from its neighbor nodes. Sharding configuration may be inferred from
//  1) Parsing XlaSharding attribute from neighboring node.
//  2) If argument node is a resource, then by parsing adjacent nodes
//     of the connected ReadVariable op.
absl::Status ParseAndValidateShardingFromNeighbors(
    const int num_cores_per_replica, const std::string& arg_node_name,
    const Node& neighbor_node, int64_t* inferred_core_id, bool* is_fast_mem,
    std::optional<NodeAndSharding>* result) {
  if (neighbor_node.attrs().Find(TPU_FAST_MEM_ATTR) != nullptr) {
    *is_fast_mem = true;
    VLOG(2) << "place " << neighbor_node.name() << " on fast memory because "
            << arg_node_name << " has " << TPU_FAST_MEM_ATTR << " attribute";
  }

  // XlaSharding information may be encoded on node directly connected to the
  // argument node.
  TF_ASSIGN_OR_RETURN(
      std::optional<NodeAndSharding> node_and_sharding,
      ParseInputShardingFromAdjacentNode(num_cores_per_replica, neighbor_node));
  if (node_and_sharding.has_value()) {
    TF_RETURN_IF_ERROR(ParseAndValidateSharding(
        *node_and_sharding, num_cores_per_replica, inferred_core_id, result));
    return absl::OkStatus();
  }

  // When we use variable in TPU computation, we always have a
  // XlaSharding op followed by a ReadVariableOp. As so, correctly parse
  // the users of ReadVariableOp for potential sharding configuration.
  if (neighbor_node.type_string() == "ReadVariableOp") {
    for (const Edge* e : neighbor_node.out_edges()) {
      if (e->IsControlEdge()) continue;

      if (e->dst()->attrs().Find(TPU_FAST_MEM_ATTR) != nullptr) {
        *is_fast_mem = true;
        VLOG(2) << "place " << arg_node_name << " on fast memory because "
                << e->dst()->name() << TPU_FAST_MEM_ATTR << " attribute";
      }

      TF_ASSIGN_OR_RETURN(
          std::optional<NodeAndSharding> node_and_sharding,
          ParseInputShardingFromAdjacentNode(num_cores_per_replica, *e->dst()));
      if (node_and_sharding.has_value()) {
        TF_RETURN_IF_ERROR(ParseAndValidateSharding(*node_and_sharding,
                                                    num_cores_per_replica,
                                                    inferred_core_id, result));
        return absl::OkStatus();
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

// Inputs:
//   replication_spec_string: the device to which the TPUReplicate node was
//     assigned.
//   device_set: the set of TF devices.
// Outputs:
//   tpu_compilation_device: the name of the TPU compilation device.
//   num_tpus_per_task: the number of TPUs in each task. Verifies that all tasks
//     have the same number of TPU devices.
//   tpu_devices: the TPU devices, indexed by [task][device].
static absl::Status GetTPUDeviceNames(
    const std::string& replication_spec_string, const DeviceSet& device_set,
    std::string* tpu_compilation_device, int* num_tpus_per_task,
    std::vector<std::vector<Device*>>* tpu_devices) {
  // TODO(b/110910013) GetSystemDevice parses the spec and returns the name of
  // the tpu_system device, which we replace by the cpu device. We do this
  // replacement because we want to place the TPUCompileOp (and the compile
  // assert op) explicitly on cpu devices on the same job as the tpu_system
  // device.
  DeviceNameUtils::ParsedName replication_spec;
  Device* replication_device;
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetSystemDevice(
      replication_spec_string, device_set, &replication_spec,
      &replication_device));
  *tpu_compilation_device =
      str_util::StringReplace(replication_device->name(), DEVICE_TPU_SYSTEM,
                              DEVICE_CPU, /*replace_all=*/true);

  // Finds the set of TPU devices attached to the tasks in the job.
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetTPUDevices(
      replication_spec, device_set, num_tpus_per_task, tpu_devices));

  return absl::OkStatus();
}

// Parses the topology attribute of TPUReplicate, and populates *topology with
// a physical mesh coordinate to (task, device) mapping.
static absl::Status ParseTopologyAttr(
    const std::string& topology_attr,
    const tpu::TpuTopologyExternal& tpu_topology, int num_tasks,
    int num_tpus_per_task, xla::Array4D<std::pair<int, int>>* topology) {
  static_assert(4 == kTPUTopologyRank, "Assumes the topology rank is 4");
  tpu::TopologyProto proto;
  proto.ParseFromString(topology_attr);
  if (proto.mesh_shape_size() != kTPUTopologyRank) {
    return absl::InvalidArgumentError(
        absl::StrCat("TPU topology must be rank ", kTPUTopologyRank));
  }
  if (proto.num_tasks() != num_tasks) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched number of TPU tasks (", proto.num_tasks(),
                     " != ", num_tasks, ")"));
  }
  if (proto.num_tpu_devices_per_task() != num_tpus_per_task) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Mismatched number of TPUs per task (",
        proto.num_tpu_devices_per_task(), " != ", num_tpus_per_task, ")."));
  }
  if (proto.device_coordinates_size() !=
      num_tasks * num_tpus_per_task * kTPUTopologyRank) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device coordinates should be ", num_tasks, "x", num_tpus_per_task, "x",
        kTPUTopologyRank, "; got ", proto.device_coordinates_size()));
  }

  int devices_per_chip = tpu_topology.LogicalDevicesPerChip(kTensorCore);
  *topology = xla::Array4D<std::pair<int, int>>(
      tpu_topology.chip_bounds().x, tpu_topology.chip_bounds().y,
      tpu_topology.chip_bounds().z, devices_per_chip, {-1, -1});
  int pos = 0;
  for (int task = 0; task < num_tasks; ++task) {
    for (int device = 0; device < num_tpus_per_task; ++device) {
      int32_t x = proto.device_coordinates(pos++);
      int32_t y = proto.device_coordinates(pos++);
      int32_t z = proto.device_coordinates(pos++);
      int32_t core = proto.device_coordinates(pos++);

      if (!tpu_topology.HasChip(x, y, z) || core < 0 ||
          core >= devices_per_chip) {
        return absl::InvalidArgumentError(
            absl::StrCat("Mesh coordinates (", x, ",", y, ",", z, ",", core,
                         ") are not valid for the current TPU topology"));
      }
      if ((*topology)(x, y, z, core).first != -1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate coordinates (", x, ",", y, ",", z, ",",
                         core, ") in TPU topology"));
      }
      (*topology)(x, y, z, core) = {task, device};
    }
  }
  return absl::OkStatus();
}

// Parses the value of the device_assignment attribute to TPUReplicate.
// Populates *device_assignment; *device_assignment must be a 2D array with
// shape (num_replicas, num_cores_per_replica).
static absl::Status ParseDeviceAssignmentAttr(
    absl::Span<const int> device_assignment_attr,
    const tpu::TpuTopologyExternal& tpu_topology, int num_replicas,
    int num_cores_per_replica,
    xla::Array2D<tpu::TpuCoreLocationExternal>* device_assignment) {
  static_assert(4 == kTPUTopologyRank, "Assumes the topology rank is 4");

  const int64_t device_assignment_attr_size =
      num_replicas * num_cores_per_replica * kTPUTopologyRank;
  if (device_assignment_attr.size() != device_assignment_attr_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Length of device_assignment attribute must be equal to num_replicas (",
        num_replicas, ") * num_cores_per_replica (", num_cores_per_replica,
        ") * ", kTPUTopologyRank, " got ", device_assignment_attr.size()));
  }
  for (int core : device_assignment_attr) {
    if (core < 0 || core >= kTPUMaxTopologySize) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid core number in device assignment: ", core));
    }
  }

  *device_assignment = xla::Array2D<tpu::TpuCoreLocationExternal>(
      num_replicas, num_cores_per_replica);
  int devices_per_chip = tpu_topology.LogicalDevicesPerChip(kTensorCore);
  xla::Array4D<int> replica_assignment(
      tpu_topology.chip_bounds().x, tpu_topology.chip_bounds().y,
      tpu_topology.chip_bounds().z, devices_per_chip, -1);
  int pos = 0;
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int logical_core = 0; logical_core < num_cores_per_replica;
         ++logical_core) {
      int32_t x = device_assignment_attr[pos++];
      int32_t y = device_assignment_attr[pos++];
      int32_t z = device_assignment_attr[pos++];
      int32_t core = device_assignment_attr[pos++];

      if (!tpu_topology.HasChip(x, y, z) || core < 0 ||
          core >= devices_per_chip) {
        return absl::InvalidArgumentError(
            absl::StrCat("Mesh coordinates (", x, ",", y, ",", core,
                         ") are not valid for the current TPU topology"));
      }
      tpu::TpuCoreLocationExternal core_location =
          tpu_topology.Core(kTensorCore, x, y, z, core);

      if (replica_assignment(x, y, z, core) != -1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate coordinates (", x, ",", y, ",", z, ",",
                         core, ") in TPU device assignment"));
      }
      replica_assignment(x, y, z, core) = replica;
      (*device_assignment)(replica, logical_core) = core_location;
    }
  }
  return absl::OkStatus();
}

// Builds TensorFlow device assignments for the special case of a single core
// computation that is replicated to every core in the mesh.
// LINT.IfChange
static absl::Status BuildFullMeshDeviceAssignment(
    int num_replicas, const std::vector<std::vector<Device*>>& tpu_devices,
    int num_tasks, int num_tpus_per_task,
    std::vector<std::vector<std::string>>* tf_device_assignment,
    std::vector<int>* devices_to_lock) {
  // Assign TensorFlow devices to replicas arbitrarily.
  for (int i = 0; i < num_replicas; ++i) {
    int task = i / num_tpus_per_task;
    int device = i % num_tpus_per_task;
    TF_RET_CHECK(task >= 0 && task < num_tasks);
    TF_RET_CHECK(device >= 0 && device < num_tpus_per_task);

    // We don't actually know which TF device corresponds to which physical
    // device, but it doesn't matter—they're all identical.
    (*tf_device_assignment)[i] = {tpu_devices[task][device]->name()};
    devices_to_lock->push_back(i);
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

// Builds TensorFlow device assignments for a replicated computation and convert
// device_assignment into xla_device_assignment.
static absl::Status BuildGeneralDeviceAssignment(
    int num_replicas, int num_cores_per_replica,
    const std::vector<std::vector<Device*>>& tpu_devices,
    const xla::Array2D<tpu::TpuCoreLocationExternal>& device_assignment,
    const xla::Array4D<std::pair<int, int>>& topology,
    std::vector<std::vector<std::string>>* tf_device_assignment,
    std::vector<int>* devices_to_lock,
    std::unique_ptr<xla::DeviceAssignment>* xla_device_assignment) {
  // Assign TensorFlow devices to each computation's replicas according to
  // device_assignment and 'topology'.
  *xla_device_assignment = std::make_unique<xla::DeviceAssignment>(
      num_replicas, num_cores_per_replica);
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int computation = 0; computation < num_cores_per_replica;
         ++computation) {
      const tpu::TpuCoreLocationExternal& core_location =
          device_assignment(replica, computation);

      int task;
      int device;
      std::tie(task, device) =
          topology(core_location.chip_coordinates().x,
                   core_location.chip_coordinates().y,
                   core_location.chip_coordinates().z, core_location.index());

      CHECK_LT(computation, num_cores_per_replica);
      (**xla_device_assignment)(replica, computation) = core_location.Id();

      // The communication pattern between replicas will be determined later by
      // BuildAllReduceRing.
      TF_RET_CHECK(task >= 0 && task < tpu_devices.size());
      TF_RET_CHECK(device >= 0 && device < tpu_devices[task].size());
      (*tf_device_assignment)[replica].push_back(
          tpu_devices[task][device]->name());
      devices_to_lock->push_back((task * tpu_devices[task].size()) + device);
    }
  }
  return absl::OkStatus();
}

/*static*/ absl::Status DistributedTPURewritePass::BuildDeviceAssignment(
    const tpu::TpuTopologyExternal& tpu_topology, int num_tpus_per_task,
    const std::vector<std::vector<Device*>>& tpu_devices, int num_replicas,
    int num_cores_per_replica, const std::string& topology_attr,
    absl::Span<const int> device_assignment_attr,
    std::vector<std::vector<std::string>>* tf_device_assignment,
    std::vector<int>* devices_to_lock,
    std::unique_ptr<xla::DeviceAssignment>* xla_device_assignment) {
  const int num_tasks = tpu_devices.size();
  const int num_tpu_devices = num_tasks * num_tpus_per_task;
  VLOG(2) << "num_tasks=" << num_tasks
          << " num_tpus_per_task=" << num_tpus_per_task;

  // Checks num_replicas is sane first to avoid integer overflow.
  if (num_replicas > num_tpu_devices) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested num_replicas=", num_replicas, " but there are only ",
        num_tpu_devices, " cores in the TPU topology."));
  }
  if (num_replicas * num_cores_per_replica > num_tpu_devices) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested num_replicas=", num_replicas, " with ",
        num_cores_per_replica, " cores per replica, but there are only ",
        num_tpu_devices, " cores in the TPU topology"));
  }

  tf_device_assignment->clear();
  tf_device_assignment->resize(num_replicas);

  devices_to_lock->clear();
  devices_to_lock->reserve(num_replicas * num_cores_per_replica);

  // Special case: we allow the user to omit the topology and device assignment
  // information in two cases:
  // * there is only one replica and one core per replica. In this case, we
  //   don't need to know topology information because we don't communicate with
  //   other cores.
  // * the number of replicas is equal to the number of cores in the slice. In
  //   this case, all cores are running the same program so we don't need to
  //   know which is which.
  if (topology_attr.empty()) {
    // LINT.IfChange
    if (num_replicas != 1 && num_replicas != num_tpu_devices) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TPUReplicate asked to create ", num_replicas,
          " replicas, but the number of cores in the TPU topology is ",
          num_tpu_devices,
          " and no TPU device assignment was supplied. "
          "A TPU device assignment is required if the number of replicas is "
          "not 1 or the number of cores in the topology (",
          num_tpu_devices, ")"));
    }

    if (num_cores_per_replica != 1) {
      return absl::InvalidArgumentError(
          "A TPU topology must be provided if num_cores_per_replica != 1");
    }

    if (!device_assignment_attr.empty()) {
      return absl::InvalidArgumentError(
          "A TPU topology must be provided if device_assignment_attr is "
          "non-empty");
    }

    // If there is only one replica, assign the Tensorflow computation to task 0
    // device 0, and leave the XLA device assignment empty. We don't know which
    // core this is in the TPU topology, but it doesn't matter—we don't need to
    // communicate with any other cores.
    if (num_replicas == 1) {
      (*tf_device_assignment)[0] = {tpu_devices[0][0]->name()};
      devices_to_lock->push_back(0);
      return absl::OkStatus();
    }

    // Otherwise, num_replicas is equal to the number of cores, and we build a
    // device assignment that covers the entire mesh. We do not need to know
    // the topology to do so because all cores are identical.
    return BuildFullMeshDeviceAssignment(num_replicas, tpu_devices, num_tasks,
                                         num_tpus_per_task,
                                         tf_device_assignment, devices_to_lock);
    // LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)
  }

  // Array that maps mesh coordinates to {TF task, TF TPU device #} pairs.
  xla::Array4D<std::pair<int, int>> topology;
  TF_RETURN_IF_ERROR(ParseTopologyAttr(topology_attr, tpu_topology, num_tasks,
                                       num_tpus_per_task, &topology));

  // Array that maps logical (replica, core) pairs to physical mesh coordinates.
  xla::Array2D<tpu::TpuCoreLocationExternal> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentAttr(
      device_assignment_attr, tpu_topology, num_replicas, num_cores_per_replica,
      &device_assignment));

  return BuildGeneralDeviceAssignment(
      num_replicas, num_cores_per_replica, tpu_devices, device_assignment,
      topology, tf_device_assignment, devices_to_lock, xla_device_assignment);
}

absl::Status DistributedTPURewritePass::GetComputationForTPUReplicateOp(
    const NameAttrList& function, FunctionLibraryRuntime* flr,
    Graph* computation, DataTypeVector* arg_types,
    DataTypeVector* retval_types) {
  FunctionLibraryRuntime::Handle handle;

  TF_RETURN_IF_ERROR(
      flr->Instantiate(function.name(), AttrSlice(&function.attr()), &handle));

  const FunctionBody* fbody = flr->GetFunctionBody(handle);

  CopyGraph(*fbody->graph, computation);
  *arg_types = fbody->arg_types;
  *retval_types = fbody->ret_types;
  return absl::OkStatus();
}

// Grab the InferredShape corresponding to an edge input.
static absl::Status GetEdgeShape(const GraphShapeInfo& shape_info,
                                 const Edge& edge, const InferredShape** info) {
  auto it = shape_info.find(edge.src()->name());
  if (it == shape_info.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input to replicated TPU computation is missing InferredShape: ",
        edge.src()->name()));
  }
  TF_RET_CHECK(it->second.size() > edge.src_output());
  *info = &it->second[edge.src_output()];
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::GetArgAndRetvalShapes(
    const GraphShapeInfo& shape_info, const Node& node,
    const ParameterInfo& params_info, std::vector<InferredShape>* arg_shapes,
    std::vector<InferredShape>* retval_shapes) {
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node.input_edges(&input_edges));

  // If any replica's arg shape is unknown, we will mark the computation's arg
  // shape as being unknown. If the shapes differ the TpuExecute Op will raise a
  // runtime error.
  std::vector<bool> any_replica_shape_unknown(
      params_info.NumInputsToEachReplica());
  arg_shapes->clear();
  arg_shapes->resize(params_info.NumInputsToEachReplica());
  TF_RET_CHECK(input_edges.size() == params_info.NumInputsFromHost());
  // Determines the shapes of the per-replica arguments and checks that all
  // replicas have identical shapes.
  int64_t edge_pos = 0;
  auto check_shape = [&](int input_index) -> absl::Status {
    const InferredShape* info;
    TF_RETURN_IF_ERROR(GetEdgeShape(shape_info, *input_edges[edge_pos], &info));
    ++edge_pos;

    if ((info->handle_type == DT_INVALID && !info->shape.IsFullyDefined()) ||
        (info->handle_type != DT_INVALID &&
         !info->handle_shape.IsFullyDefined())) {
      any_replica_shape_unknown[input_index] = true;
    }
    absl::StatusOr<InferredShape> status =
        MergeInferredShapes((*arg_shapes)[input_index], *info);
    if (!status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Mismatched shapes for input ", input_index, ": ",
                       (*arg_shapes)[input_index].shape.DebugString(), " vs. ",
                       info->shape.DebugString()));
    }
    (*arg_shapes)[input_index] = status.value();
    return absl::OkStatus();
  };

  for (int64_t i = 0; i < params_info.NumReplicas(); ++i) {
    for (int64_t j = 0; j < params_info.NumPerReplicaArgs(); ++j) {
      TF_RETURN_IF_ERROR(check_shape(j));
    }
  }

  for (int64_t i = 0; i < params_info.NumDistributedArgs(); ++i) {
    TF_RETURN_IF_ERROR(check_shape(params_info.NumPerReplicaArgs() + i));
  }

  for (int64_t i = 0;
       i < params_info.NumPerReplicaArgs() + params_info.NumDistributedArgs();
       ++i) {
    if (any_replica_shape_unknown[i]) {
      (*arg_shapes)[i].shape = PartialTensorShape();
      (*arg_shapes)[i].handle_shape = PartialTensorShape();
    }
  }

  // Determines the shape of the broadcast arguments.
  for (int64_t i = 0; i < params_info.NumBroadcastArgs(); ++i) {
    TF_RET_CHECK(node.input_type(edge_pos) != DT_RESOURCE);
    const InferredShape* info;
    TF_RETURN_IF_ERROR(GetEdgeShape(shape_info, *input_edges[edge_pos], &info));
    (*arg_shapes)[i + params_info.NumPerReplicaArgs() +
                  params_info.NumDistributedArgs()]
        .shape = info->shape;
    ++edge_pos;
  }

  // Determines the handle shape and handle type of the resource variable
  // arguments.
  for (int64_t i = 0; i < params_info.NumVariables(); ++i) {
    TF_RET_CHECK(node.input_type(edge_pos) == DT_RESOURCE);
    const InferredShape* info;
    TF_RETURN_IF_ERROR(GetEdgeShape(shape_info, *input_edges[edge_pos], &info));
    InferredShape& arg_shape =
        (*arg_shapes)[i + params_info.NumPerReplicaArgs() +
                      params_info.NumDistributedArgs() +
                      params_info.NumBroadcastArgs()];
    arg_shape.shape = TensorShape();  // Variables are always scalars.
    arg_shape.handle_shape = info->handle_shape;
    arg_shape.handle_type = info->handle_type;
    TF_RET_CHECK(arg_shape.handle_type != DT_INVALID)
        << " input edge: " << input_edges[edge_pos]->DebugString();
    ++edge_pos;
  }

  // Determines the shape of the guaranteed constants.
  // TODO(vinuraja): Can be removed because they are not required for any
  // calculations. Leaving them here for symmetry with other structures like
  // arg_types, arg_sharding, etc.
  for (int64_t i = 0; i < params_info.NumGuaranteedConstants(); ++i) {
    TF_RET_CHECK(node.input_type(edge_pos) != DT_RESOURCE);
    const InferredShape* info;
    TF_RETURN_IF_ERROR(GetEdgeShape(shape_info, *input_edges[edge_pos], &info));
    (*arg_shapes)[i + params_info.NumPerReplicaArgs() +
                  params_info.NumDistributedArgs() +
                  params_info.NumBroadcastArgs() + params_info.NumVariables()]
        .shape = info->shape;
    ++edge_pos;
  }

  // Extract the return value shapes.
  auto it = shape_info.find(node.name());
  retval_shapes->clear();
  if (it != shape_info.end()) {
    TF_RET_CHECK(it->second.size() >= node.num_outputs());
    retval_shapes->resize(node.num_outputs());
    for (int i = 0; i < node.num_outputs(); ++i) {
      (*retval_shapes)[i].shape = it->second[i].shape;
    }
  } else if (node.num_outputs() > 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Replicated TPU computation is missing InferredShape: ",
                     FormatNodeForError(node)));
  }
  return absl::OkStatus();
}

// Verifies that all nodes have legal sharding.
static absl::Status ValidateCoreNumbers(const Graph& graph,
                                        int num_cores_per_replica) {
  for (Node* n : graph.nodes()) {
    TF_ASSIGN_OR_RETURN(std::optional<xla::OpSharding> sharding,
                        ParseShardingFromDevice(*n, num_cores_per_replica,
                                                /*add_metadata=*/true));
  }
  return absl::OkStatus();
}

static absl::Status InferXlaShardingFromNeighbors(
    const Node& n, int num_cores_per_replica, FunctionLibraryRuntime* flr,
    CachedFunctionHandles* cached_function_handles,
    std::optional<NodeAndSharding>* output_node_and_sharding,
    bool* is_fast_mem) {
  int64_t core = -1;
  std::optional<NodeAndSharding> result;
  // We assume the variable has been allocated on fast memory if any consuming
  // op has TPU_FAST_MEM_ATTR attribute. This is a protocol between runtime and
  // compiler.
  *is_fast_mem = false;
  for (const Edge* edge : n.out_edges()) {
    if (edge->IsControlEdge()) continue;

    TF_RETURN_IF_ERROR(ParseAndValidateShardingFromNeighbors(
        num_cores_per_replica, n.name(), *edge->dst(), &core, is_fast_mem,
        &result));

    if (!flr) continue;

    // The nodes deciding this arg's device assignment might be in
    // FunctionDef. Instantiate FunctionDefs associated with this node
    // and check nodes using this arg.
    std::function<absl::Status(const Edge* call_edge)>
        parse_sharding_from_function = [&](const Edge* call_edge) {
          auto associated_functions = GetAssociatedFunctions(
              *call_edge->dst(), flr->GetFunctionLibraryDefinition());
          for (auto& associated_function : associated_functions) {
            FunctionLibraryRuntime::Handle handle;
            TF_RETURN_IF_ERROR(cached_function_handles->GetOrInstantiate(
                associated_function.func_name(),
                AttrSlice(&associated_function.attrs()), &handle));
            const FunctionBody* body = flr->GetFunctionBody(handle);
            Graph* g = body->graph;

            for (Node* body_node : g->nodes()) {
              if (!body_node->IsArg()) continue;

              int index;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(body_node->attrs(), "index", &index));
              if (index != call_edge->dst_input()) continue;

              for (const Edge* out_edge : body_node->out_edges()) {
                if (out_edge->IsControlEdge()) continue;

                TF_RETURN_IF_ERROR(ParseAndValidateShardingFromNeighbors(
                    num_cores_per_replica, n.name(), *out_edge->dst(), &core,
                    is_fast_mem, &result));

                TF_RETURN_IF_ERROR(parse_sharding_from_function(out_edge));
              }
            }
          }
          return absl::OkStatus();
        };
    TF_RETURN_IF_ERROR(parse_sharding_from_function(edge));
  }
  *output_node_and_sharding = result;
  return absl::OkStatus();
}

bool UseSpmdForXlaPartitioning(const Node* replicate_node) {
  bool spmd_attr;
  if (!replicate_node ||
      !TryGetNodeAttr(replicate_node->attrs(), "use_spmd_for_xla_partitioning",
                      &spmd_attr)) {
    spmd_attr = false;
  }
  return spmd_attr;
}

std::string FormatNodeAndShardingMsg(
    const std::optional<NodeAndSharding>& node_and_sharding) {
  DCHECK(node_and_sharding.has_value());

  xla::OpSharding sharding_no_metadata = node_and_sharding->sharding;
  sharding_no_metadata.clear_metadata();
  std::string escaped_sharding_str =
      absl::CEscape(sharding_no_metadata.SerializeAsString());
  if (node_and_sharding->node == nullptr) {
    return absl::StrCat(" via default sharding '", escaped_sharding_str, "'");
  }

  return absl::StrCat(" via node ", node_and_sharding->node->DebugString(),
                      " sharding '", escaped_sharding_str, "'");
}

absl::Status DistributedTPURewritePass::AssignArgsAndRetvalsToCores(
    int num_cores_per_replica, const ParameterInfo& params_info,
    const DataTypeVector& arg_types,
    const std::vector<InferredShape>& arg_shapes,
    const DataTypeVector& retval_types,
    const std::vector<InferredShape>& retval_shapes, const Graph& graph,
    const Node* replicate_node, FunctionLibraryRuntime* flr,
    bool allow_parameter_replication_for_spmd,
    std::vector<xla::OpSharding>* arg_sharding, std::vector<bool>* arg_fast_mem,
    std::vector<xla::OpSharding>* retval_sharding,
    std::vector<std::string>* arg_names) {
  // Builds vectors of the argument and return nodes.
  std::vector<Node*> args(arg_types.size());
  std::vector<Node*> retvals(retval_types.size());
  absl::flat_hash_map<int, Node*> partitioned_output_nodes;
  for (Node* node : graph.op_nodes()) {
    if (node->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < args.size());
      args[index] = node;
    } else if (node->IsRetval()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < retvals.size());
      retvals[index] = node;
    }
  }
  for (const Edge* edge : replicate_node->out_edges()) {
    int num_partitioned_outputs = 0;
    for (const Edge* out_edge : edge->dst()->out_edges()) {
      if (_IsTPUPartitionedOutput(out_edge->dst())) {
        partitioned_output_nodes[edge->src_output()] = out_edge->dst();
        num_partitioned_outputs++;
      }
    }
    if (num_partitioned_outputs > 1) {
      return absl::InvalidArgumentError(
          "More than one TPUPartitionedOutput per replciated output.");
    }
  }

  // Verifies there are no missing arguments/return values.
  for (int i = 0; i < args.size(); ++i) {
    if (args[i] == nullptr) {
      return absl::InternalError(
          absl::StrCat("Missing function argument: ", i));
    }
  }
  for (int i = 0; i < retvals.size(); ++i) {
    if (retvals[i] == nullptr) {
      return absl::InternalError(
          absl::StrCat("Missing function return value: ", i));
    }
  }

  // Assigns a core to each _Arg. Chooses the lowest-numbered core that
  // consumes the argument. We choose the lowest-numbered core so the
  // assignment is deterministic.
  TensorDevicePlacer args_device_selector(num_cores_per_replica, arg_types,
                                          arg_shapes);
  arg_sharding->resize(args.size());
  arg_names->resize(args.size());
  arg_fast_mem->resize(args.size());
  CachedFunctionHandles cached_function_handles(flr);
  const bool use_spmd = (UseSpmdForXlaPartitioning(replicate_node) ||
                         replicate_inputs_outputs_by_default_for_xla_spmd_) &&
                        allow_parameter_replication_for_spmd &&
                        num_cores_per_replica > 1;

  // Offset _TPUReplicate non per replica argument indices by
  // (num_replicas - 1) * num_per_replica_args as _TPUReplicate nodes are
  // constructed with all per replica args across all replicas while the
  // encapsulated function only has 1 replica's per replica args. Per replica
  // args are ordered by replica first, so the index here does not require an
  // offset and the first replica's input nodes is sufficient for determining
  // argument sharding.
  const int index_offset =
      (params_info.NumReplicas() - 1) * params_info.NumPerReplicaArgs();
  for (int i = 0; i < args.size(); ++i) {
    const Node* n = args[i];
    std::optional<int64_t> assigned_core;
    std::optional<NodeAndSharding> node_and_sharding;
    bool is_fast_mem;
    TF_RETURN_IF_ERROR(InferXlaShardingFromNeighbors(
        *n, num_cores_per_replica, flr, &cached_function_handles,
        &node_and_sharding, &is_fast_mem));

    const bool is_per_replica_arg = params_info.IsPerReplicaArg(i);
    if (is_per_replica_arg || params_info.IsDistributedArg(i)) {
      Node* input_node;
      TF_RETURN_IF_ERROR(replicate_node->input_node(
          i + (is_per_replica_arg ? 0 : index_offset), &input_node));
      if (_IsTPUPartitionedInput(input_node)) {
        TF_ASSIGN_OR_RETURN(
            std::optional<xla::OpSharding> parsed_sharding,
            GetShardingFromNodeDef(input_node->def(), /*add_metadata=*/true));
        if (!parsed_sharding.has_value())
          return absl::InvalidArgumentError(absl::StrCat(
              "Missing _XlaSharding attr from: ", input_node->DebugString()));
        node_and_sharding = NodeAndSharding(input_node, *parsed_sharding);
        VLOG(1) << "Arg " << i << " parsed sharding information from "
                << input_node->DebugString() << " : "
                << parsed_sharding->DebugString();
      }
    }

    if (params_info.IsVariableArg(i)) {
      Node* input_node;
      TF_RETURN_IF_ERROR(
          replicate_node->input_node(i + index_offset, &input_node));
      if (input_node->type_string() == kVarHandleOp) {
        TF_ASSIGN_OR_RETURN(
            std::optional<xla::OpSharding> parsed_sharding,
            GetShardingFromNodeDef(input_node->def(), /*add_metadata=*/true));
        if (parsed_sharding.has_value()) {
          node_and_sharding = NodeAndSharding(input_node, *parsed_sharding);
          VLOG(1) << "Arg " << i << " parsed sharding information from "
                  << input_node->DebugString() << " : "
                  << parsed_sharding->DebugString();
        }
      }
    }

    if (node_and_sharding.has_value() && enable_automatic_model_parallelism_) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Specifying manual sharding is not allowed when automatic "
          "model parallelism is enabled.",
          node_and_sharding->sharding.DebugString()));
    }

    if (!node_and_sharding.has_value()) {
      if (use_spmd &&
          (params_info.IsVariableArg(i) || params_info.IsBroadcastArg(i) ||
           ((params_info.IsPerReplicaArg(i) ||
             params_info.IsDistributedArg(i)) &&
            arg_types[i] != DT_RESOURCE) ||
           params_info.IsConstantArg(i))) {
        // Use replication for host variables or non-variable per-replica
        // inputs.
        node_and_sharding = NodeAndSharding(/*node=*/nullptr,
                                            xla::sharding_builder::Replicate());
      } else {
        // TODO(dlibenzi): Distributing variables to cores other than 0 makes
        // learning/brain/research/babelfish/trainer:trainer_tpu_test fail.
        // For now distribute only per replica arguments, unless
        // tf_jf_distribute_vars is set, to allow debugging the issue.
        if (((params_info.IsPerReplicaArg(i) ||
              params_info.IsDistributedArg(i)) &&
             arg_types[i] != DT_RESOURCE) ||
            (distribute_vars_ && params_info.IsVariableArg(i))) {
          assigned_core = args_device_selector.RetrieveAssignment(i);
        } else {
          assigned_core = 0;
        }
        node_and_sharding = NodeAndSharding(
            /*node=*/nullptr,
            xla::sharding_builder::AssignDevice(*assigned_core));
      }
      *node_and_sharding->sharding.add_metadata() =
          CreateOpMetadataFromNode(*replicate_node);
    } else if (node_and_sharding->sharding.type() == xla::OpSharding::MAXIMAL) {
      if (use_spmd) {
        node_and_sharding->sharding = xla::sharding_builder::Replicate();
      } else {
        assigned_core = node_and_sharding->sharding.tile_assignment_devices(0);
      }
    } else if (node_and_sharding->sharding.type() !=
                   xla::OpSharding::REPLICATED &&
               node_and_sharding->sharding.type() != xla::OpSharding::OTHER) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported argument sharding (for arg ", n->DebugString(),
          "): ", node_and_sharding->sharding.DebugString()));
    }
    if (assigned_core.has_value()) {
      args_device_selector.ReportDeviceAssigned(*assigned_core, i);
      VLOG(3) << "Assigning argument " << i << " (" << n->DebugString()
              << ") to core " << *assigned_core
              << FormatNodeAndShardingMsg(node_and_sharding);
      args[i]->set_assigned_device_name(CoreDeviceLabel(*assigned_core));
    } else if (node_and_sharding->sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t core :
           node_and_sharding->sharding.tile_assignment_devices()) {
        TF_RET_CHECK(core >= 0 && core < num_cores_per_replica)
            << "core " << core << " should be between [0, "
            << num_cores_per_replica << "). sharding is "
            << node_and_sharding->sharding.DebugString();
        args_device_selector.ReportDeviceAssigned(core, i);
      }
      VLOG(3) << "Assigning argument " << i << " (" << n->DebugString()
              << ") with tiled sharding to cores "
              << absl::StrJoin(
                     node_and_sharding->sharding.tile_assignment_devices(), ",")
              << " " << FormatNodeAndShardingMsg(node_and_sharding);
    } else {
      DCHECK_EQ(node_and_sharding->sharding.type(),
                xla::OpSharding::REPLICATED);
      for (int64_t core = 0; core < num_cores_per_replica; ++core) {
        args_device_selector.ReportDeviceAssigned(core, i);
      }
      VLOG(3) << "Assigning argument " << i << " (" << n->DebugString()
              << ") to all cores"
              << FormatNodeAndShardingMsg(node_and_sharding);
    }
    (*arg_sharding)[i] = node_and_sharding->sharding;
    (*arg_fast_mem)[i] = is_fast_mem;
    (*arg_names)[i] = n->name();
    if (is_fast_mem) {
      VLOG(3) << "Add " << TPU_FAST_MEM_ATTR << " attribute to "
              << args[i]->name();
    }
    args[i]->AddAttr(kShardingAttribute,
                     node_and_sharding->sharding.SerializeAsString());
  }
  TF_RETURN_IF_ERROR(cached_function_handles.ReleaseAllHandles());

  // Assigns each _Retval node to the core that produces its value.
  TensorDevicePlacer retvals_device_selector(num_cores_per_replica,
                                             retval_types, retval_shapes);
  retval_sharding->resize(retvals.size());
  for (int i = 0; i < retvals.size(); ++i) {
    const Edge* edge;
    TF_RETURN_IF_ERROR(retvals[i]->input_edge(0, &edge));

    TF_ASSIGN_OR_RETURN(
        std::optional<xla::OpSharding> edge_sharding,
        ParseShardingFromEdgeSource(*edge, num_cores_per_replica,
                                    /*add_metadata=*/true));

    std::optional<NodeAndSharding> node_and_sharding;
    if (edge_sharding.has_value()) {
      node_and_sharding.emplace(NodeAndSharding(edge->src(), *edge_sharding));
    }

    if (partitioned_output_nodes.contains(i)) {
      Node* output_node = partitioned_output_nodes[i];
      TF_ASSIGN_OR_RETURN(
          std::optional<xla::OpSharding> parsed_sharding,
          GetShardingFromNodeDef(output_node->def(), /*add_metadata=*/true));
      if (parsed_sharding.has_value()) {
        node_and_sharding = NodeAndSharding(output_node, *parsed_sharding);
        VLOG(1) << "Retval " << i << " parsed sharding information from "
                << output_node->DebugString() << " : "
                << parsed_sharding->DebugString();
      }
    }
    std::optional<int64_t> assigned_core;
    if (node_and_sharding.has_value()) {
      if (enable_automatic_model_parallelism_) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Specifying manual sharding is not allowed when automatic "
            "model parallelism is enabled.",
            node_and_sharding->sharding.DebugString()));
      }

      if (node_and_sharding->sharding.type() == xla::OpSharding::MAXIMAL) {
        if (use_spmd) {
          node_and_sharding->sharding = xla::sharding_builder::Replicate();
        } else {
          assigned_core =
              node_and_sharding->sharding.tile_assignment_devices(0);
          TF_RETURN_IF_ERROR(
              ValidateCoreNumber(*assigned_core, num_cores_per_replica));
        }
      } else if (node_and_sharding->sharding.type() !=
                     xla::OpSharding::REPLICATED &&
                 node_and_sharding->sharding.type() != xla::OpSharding::OTHER) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported argument sharding for retval ",
            retvals[i]->DebugString(), " edge=", edge->DebugString(), ": ",
            node_and_sharding->sharding.DebugString()));
      }
    } else {
      if (use_spmd) {
        node_and_sharding = NodeAndSharding(/*node=*/nullptr,
                                            xla::sharding_builder::Replicate());
      } else {
        if (distribute_vars_) {
          assigned_core = retvals_device_selector.RetrieveAssignment(i);
        } else {
          assigned_core = 0;
        }
        node_and_sharding = NodeAndSharding(
            /*node=*/nullptr,
            xla::sharding_builder::AssignDevice(*assigned_core));
      }
      *node_and_sharding->sharding.add_metadata() =
          CreateOpMetadataFromNode(*replicate_node);
    }
    if (assigned_core.has_value() && !use_spmd) {
      retvals[i]->set_assigned_device_name(CoreDeviceLabel(*assigned_core));
      retvals_device_selector.ReportDeviceAssigned(*assigned_core, i);
      VLOG(3) << "Assigning return value " << i << " ("
              << retvals[i]->DebugString() << ") to core " << *assigned_core
              << FormatNodeAndShardingMsg(node_and_sharding);
    } else if (node_and_sharding->sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t core :
           node_and_sharding->sharding.tile_assignment_devices()) {
        TF_RET_CHECK(core >= 0 && core < num_cores_per_replica)
            << "core " << core << " should be between [0, "
            << num_cores_per_replica << "). sharding is "
            << node_and_sharding->sharding.DebugString();
        retvals_device_selector.ReportDeviceAssigned(core, i);
      }
      VLOG(3) << "Assigning return value " << i << " ("
              << retvals[i]->DebugString() << ") with tiled sharding to cores "
              << absl::StrJoin(
                     node_and_sharding->sharding.tile_assignment_devices(), ",")
              << " " << FormatNodeAndShardingMsg(node_and_sharding);
    } else {
      if (use_spmd) {
        node_and_sharding->sharding = xla::sharding_builder::Replicate();
      }
      for (int64_t core = 0; core < num_cores_per_replica; ++core) {
        retvals_device_selector.ReportDeviceAssigned(core, i);
      }
      VLOG(3) << "Assigning return value " << i << " ("
              << retvals[i]->DebugString() << ") to all cores"
              << FormatNodeAndShardingMsg(node_and_sharding);
    }
    retvals[i]->AddAttr(kShardingAttribute,
                        node_and_sharding->sharding.SerializeAsString());
    (*retval_sharding)[i] = node_and_sharding->sharding;
  }
  if (use_spmd &&
      (absl::c_any_of(*arg_sharding,
                      [](const xla::OpSharding& s) {
                        return s.type() == xla::OpSharding::MAXIMAL;
                      }) ||
       absl::c_any_of(*retval_sharding, [](const xla::OpSharding& s) {
         return s.type() == xla::OpSharding::MAXIMAL;
       }))) {
    return absl::InvalidArgumentError(
        "XLA SPMD only supports cases where all inputs/outputs "
        "exist on every partition (sharded or replicated).");
  }
  return absl::OkStatus();
}

// Builds Shape nodes that compute the shapes of arguments whose shapes are not
// statically known.
/* static */ absl::Status DistributedTPURewritePass::BuildDynamicShapeNodes(
    const Node& replicate_node, const std::vector<InferredShape>& arg_shapes,
    const ParameterInfo& params_info, const std::vector<Node*>& variable_reads,
    Graph* graph, std::vector<Node*>* dynamic_shape_nodes) {
  dynamic_shape_nodes->clear();

  std::vector<const Edge*> replicate_input_edges;
  TF_RETURN_IF_ERROR(replicate_node.input_edges(&replicate_input_edges));

  // The compiler determines the shape of each constant by inspecting the value
  // of its corresponding host-memory tensor; this happens when a step is run.
  // As a result, the shapes of constants are not needed at graph rewrite time.
  const int num_args = arg_shapes.size() - params_info.NumGuaranteedConstants();
  TF_RET_CHECK(num_args == params_info.NumPerReplicaArgs() +
                               params_info.NumDistributedArgs() +
                               params_info.NumBroadcastArgs() +
                               params_info.NumVariables());

  for (int i = 0; i < num_args; ++i) {
    const PartialTensorShape* shape = arg_shapes[i].handle_type == DT_INVALID
                                          ? &arg_shapes[i].shape
                                          : &arg_shapes[i].handle_shape;
    if (!shape->IsFullyDefined()) {
      NodeDef def;
      Node* src;
      int src_output;
      std::vector<Node*> control_inputs;

      if (params_info.IsVariableArg(i)) {
        int64_t var_num = i - params_info.NumPerReplicaArgs() -
                          params_info.NumDistributedArgs() -
                          params_info.NumBroadcastArgs();
        TF_RET_CHECK(0 <= var_num && var_num < variable_reads.size());
        Node* read = variable_reads[var_num];

        DCHECK_EQ(read->type_string(), "ReadVariableOp");

        for (const Edge* edge : read->in_edges()) {
          if (edge->IsControlEdge()) {
            control_inputs.push_back(edge->src());
          }
        }

        const Edge* variable_input = nullptr;
        TF_RETURN_IF_ERROR(read->input_edge(/*idx=*/0, &variable_input));
        src = variable_input->src();
        src_output = variable_input->src_output();

        def.set_name(
            graph->NewName(absl::StrCat(src->name(), "/variable_shape")));
        def.set_op("VariableShape");
      } else {
        if (params_info.IsPerReplicaArg(i)) {
          TF_RET_CHECK(i < replicate_input_edges.size());
          // All replicas must have the same input shapes. Uses the shape of the
          // inputs from the first replica.
          src = replicate_input_edges[i]->src();
          src_output = replicate_input_edges[i]->src_output();
        } else {
          DCHECK(params_info.IsDistributedArg(i) ||
                 params_info.IsBroadcastArg(i));
          int64_t input_num =
              params_info.NumPerReplicaArgs() * params_info.NumReplicas() + i -
              params_info.NumPerReplicaArgs();
          TF_RET_CHECK(0 <= input_num &&
                       input_num < replicate_input_edges.size());
          src = replicate_input_edges[input_num]->src();
          src_output = replicate_input_edges[input_num]->src_output();
        }

        def.set_name(graph->NewName(absl::StrCat(src->name(), "/shape")));
        def.set_op("Shape");
        AddNodeAttr("T", src->output_type(src_output), &def);
      }

      def.set_device(src->assigned_device_name());
      AddNodeAttr("out_type", DT_INT64, &def);
      MergeDebugInfo(NodeDebugInfo(replicate_node.def()), &def);

      TF_ASSIGN_OR_RETURN(Node * shape_node, graph->AddNode(def));
      dynamic_shape_nodes->push_back(shape_node);

      shape_node->set_assigned_device_name(src->assigned_device_name());
      graph->AddEdge(src, src_output, shape_node, 0);
      for (Node* control_input : control_inputs) {
        graph->AddControlEdge(control_input, shape_node);
      }
    }
  }
  return absl::OkStatus();
}

namespace {

bool XlaBroadcastTypeSupported(const DataType dtype) {
  // Supported data types: types that map to XLA types that are <= 4 bytes.
  xla::PrimitiveType xla_type;
  auto status_or_type = DataTypeToPrimitiveType(dtype, &xla_type);
  if (!status_or_type.ok()) {
    return false;
  }
  return xla::ShapeUtil::ByteSizeOfPrimitiveType(xla_type) <= 4;
}

bool XlaBroadcastKindSupported(
    const DistributedTPURewritePass::ParameterInfo& params_info,
    int param_num) {
  // NOTE: This is intended to cover non-sharded data parallel variables, for
  // training only. . Is it correct to just check if the arg_type is
  // DT_RESOURCE?
  return params_info.IsVariableArg(param_num) &&
         !(params_info.IsPerReplicaArg(param_num) ||
           params_info.IsDistributedArg(param_num) ||
           params_info.IsBroadcastArg(param_num) ||
           params_info.IsConstantArg(param_num));
}

bool EnableXlaParamBroadcast(
    bool enable_xla_param_broadcast, bool mpmd,
    const DistributedTPURewritePass::ParameterInfo& params_info, int param_num,
    DataType dtype) {
  // Conditions necessary to use XLA collectives for arg broadcast:
  // 1. Globally enabled via enable_xla_param_broadcast.
  // 2. DataType must be supported.
  // 3. Parameter must be a variable, and not distributed or broadcasted.
  // 4. For multi-core models (num_cores_per_replica > 1), must use SPMD.
  return enable_xla_param_broadcast && XlaBroadcastTypeSupported(dtype) &&
         XlaBroadcastKindSupported(params_info, param_num) && !mpmd;
}

}  // namespace

// Builds a TPUCompile node that compiles the bodies of the function call
// `nodes`.
absl::Status DistributedTPURewritePass::BuildCompileNode(
    const Node* replicate_node, const NameAttrList& function,
    uint64_t library_fingerprint, const ParameterInfo& params_info,
    const std::vector<InferredShape>& arg_shapes,
    const DataTypeVector& arg_types,
    const std::vector<Node*>& guaranteed_constant_nodes,
    const std::string& session_handle,
    const std::vector<xla::OpSharding>& arg_sharding,
    const std::vector<bool>& arg_fast_mem,
    const std::vector<std::string>& arg_names,
    const std::vector<xla::OpSharding>& retval_sharding,
    int num_cores_per_replica, const std::string& compile_device,
    const xla::DeviceAssignment* xla_device_assignment,
    const std::vector<Node*>& dynamic_shape_nodes, Graph* graph,
    Node** compile_node, int64_t autotuner_thresh) {
  VLOG(1) << "BuildCompileNode";

  tpu::TPUCompileMetadataProto proto;
  if (replicate_node) {
    std::string str;
    TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node->attrs(),
                                   "tpu_compile_options_proto", &str));
    TF_RET_CHECK(proto.mutable_compile_options()->ParseFromString(str));
  }
  proto.set_num_replicas(params_info.NumReplicas());
  proto.set_num_cores_per_replica(num_cores_per_replica);
  proto.set_function_library_fingerprint(library_fingerprint);
  proto.set_enable_automatic_model_parallelism(
      enable_cross_replica_sharding_mirrored_variables_);
  const bool use_spmd =
      UseSpmdForXlaPartitioning(replicate_node) && allow_xla_spmd_partition_;
  proto.set_use_spmd_for_xla_partitioning(use_spmd);
  const bool mpmd = (num_cores_per_replica > 1) && !use_spmd;

  // Get and fill padding map.
  if (replicate_node != nullptr) {
    xla::DebugOptions::StepMarkerLocation location;
    TF_RETURN_IF_ERROR(GetStepMarkerLocation(*replicate_node, &location));
    proto.set_step_marker_location(location);
  }

  if (xla_device_assignment != nullptr) {
    xla_device_assignment->Serialize(proto.mutable_device_assignment());
  }

  const int num_args = arg_types.size();
  const int num_guaranteed_constants = guaranteed_constant_nodes.size();
  const int guaranteed_const_start_index = num_args - num_guaranteed_constants;
  TF_RET_CHECK(num_args == arg_shapes.size());
  TF_RET_CHECK(num_args == arg_sharding.size())
      << num_args << " != " << arg_sharding.size();

  for (int i = 0; i < num_args; ++i) {
    tpu::TPUCompileMetadataProto::Arg* arg = proto.add_args();
    DataType type = arg_types[i];
    const InferredShape& arg_shape = arg_shapes[i];
    arg->set_name(arg_names[i]);
    if (type == DT_RESOURCE) {
      TF_RET_CHECK(arg_shape.handle_type != DT_INVALID) << i;
      arg->set_dtype(arg_shape.handle_type);
      arg_shape.handle_shape.AsProto(arg->mutable_shape());
      arg->set_kind(tpu::TPUCompileMetadataProto::Arg::VARIABLE);
      arg->set_fast_mem(arg_fast_mem[i]);
    } else {
      arg->set_dtype(type);
      arg_shape.shape.AsProto(arg->mutable_shape());
      if (i >= guaranteed_const_start_index) {
        const DataType edge_type =
            guaranteed_constant_nodes[i - guaranteed_const_start_index]
                ->output_type(0);
        TF_RET_CHECK(type == edge_type)
            << "Arg type: " << type << " but edge type: " << edge_type;
        arg->set_kind(tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT);
      } else {
        arg->set_kind(tpu::TPUCompileMetadataProto::Arg::PARAMETER);
      }
    }

    // Use XLA collective primitives to distribute variables to all replicas.
    arg->set_requires_xla_broadcast(
        params_info.NumReplicas() > 1 &&
        EnableXlaParamBroadcast(enable_xla_param_broadcast_, mpmd, params_info,
                                i, arg_shape.handle_type));

    // As long as the argument is not a per-replica one, it should have the same
    // value for all replicas. For clarity, we keep the (redundant) checks for
    // variable, broadcast and constant types, to prevent bugs in case new types
    // with different semantics are introduced in the future.
    arg->set_is_same_data_across_replicas(
        !params_info.IsPerReplicaArg(i) && !params_info.IsDistributedArg(i) &&
        (params_info.IsVariableArg(i) || params_info.IsBroadcastArg(i) ||
         params_info.IsConstantArg(i)));
    if (params_info.mirrored_variable_indices().count(i) > 0) {
      TF_RET_CHECK(type == DT_RESOURCE)
          << "Arg type: " << type << " name: " << arg->name()
          << " shape: " << arg->shape().DebugString();
      arg->set_is_same_data_across_replicas(true);
      // 64-bit type is not shardable by XLA:TPU yet.
      bool sharding_enabled = (arg_shape.handle_type != DT_COMPLEX64 &&
                               arg_shape.handle_type != DT_INT64 &&
                               arg_shape.handle_type != DT_UINT64 &&
                               arg_shape.handle_type != DT_DOUBLE);
      arg->set_enable_xla_sharding(
          sharding_enabled ? tpu::TPUCompileMetadataProto::Arg::TENTATIVE
                           : tpu::TPUCompileMetadataProto::Arg::DISALLOWED);
    }
    *arg->mutable_sharding() = arg_sharding[i];
  }

  const int num_retvals = retval_sharding.size();
  for (int i = 0; i < num_retvals; ++i) {
    *proto.add_retvals()->mutable_sharding() = retval_sharding[i];
  }
  proto.set_session_handle(session_handle);

  DataTypeVector constant_arg_types;
  constant_arg_types.reserve(num_guaranteed_constants);
  for (int i = 0; i < num_guaranteed_constants; ++i) {
    constant_arg_types.push_back(arg_types[guaranteed_const_start_index + i]);
  }
  proto.set_xla_fusion_autotuner_thresh(autotuner_thresh);

  std::string metadata;
  proto.SerializeToString(&metadata);

  NodeDef def;
  def.set_name(UniqueNodeName("TPUReplicate/_compile", graph));
  def.set_op("TPUCompile");
  def.set_device(compile_device);
  if (replicate_node) {
    MergeDebugInfo(NodeDebugInfo(replicate_node->def()), &def);
  }

  AddNodeAttr("function", function, &def);
  AddNodeAttr("num_computations", num_cores_per_replica, &def);
  AddNodeAttr("NumDynamicShapes", static_cast<int>(dynamic_shape_nodes.size()),
              &def);
  AddNodeAttr("metadata", metadata, &def);
  AddNodeAttr("Tguaranteed_constants", constant_arg_types, &def);

  TF_ASSIGN_OR_RETURN(*compile_node, graph->AddNode(def));

  (*compile_node)->set_assigned_device_name(compile_device);

  for (int i = 0; i < dynamic_shape_nodes.size(); ++i) {
    graph->AddEdge(dynamic_shape_nodes[i], 0, *compile_node, i);
  }

  for (int i = 0; i < num_guaranteed_constants; ++i) {
    graph->AddEdge(guaranteed_constant_nodes[i], 0, *compile_node,
                   dynamic_shape_nodes.size() + i);
  }
  VLOG(1) << "BuildCompileNode()";
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::FindGuaranteedConstantInputs(
    const Node& node, const NameRangeMap& input_range_map,
    std::vector<Node*>* guaranteed_constants) {
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node.input_edges(&input_edges));
  std::pair<int, int> variables_limits =
      input_range_map.at("guaranteed_constants");
  for (int i = variables_limits.first; i < variables_limits.second; ++i) {
    guaranteed_constants->push_back(input_edges[i]->src());
  }
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::FindVariableInputs(
    const Node& node, const NameRangeMap& input_range_map,
    std::vector<VariableInput>* variables) {
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node.input_edges(&input_edges));
  std::pair<int, int> variables_limits = input_range_map.at("variables");
  for (int i = variables_limits.first; i < variables_limits.second; ++i) {
    Node* node = input_edges[i]->src();

    // Find the type of the VarHandleOp that feeds this node, looking through
    // any wrapping Enter or Switch nodes.
    while (node->IsEnter() || node->IsSwitch()) {
      TF_RETURN_IF_ERROR(node->input_node(0, &node));
    }
    // Fix the variable device assignment if it is requested with a full name.
    if (!node->has_assigned_device_name() &&
        !node->requested_device().empty()) {
      DeviceNameUtils::ParsedName var_device;
      TF_RET_CHECK(DeviceNameUtils::ParseFullName(node->requested_device(),
                                                  &var_device));
      if (var_device.has_job && var_device.has_replica && var_device.has_task &&
          var_device.has_type && var_device.has_id) {
        node->set_assigned_device_name(node->requested_device());
        if (node != input_edges[i]->src() &&
            !input_edges[i]->src()->has_assigned_device_name()) {
          input_edges[i]->src()->set_assigned_device_name(
              node->requested_device());
        }
      }
    }
    if (node->type_string() == kVarHandleOp) {
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));
      variables->push_back(VariableInput{input_edges[i]->src(),
                                         input_edges[i]->src_output(), dtype});
    } else if (node->type_string() == "_Arg") {
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "_handle_dtypes", &dtypes));
      if (dtypes.empty()) {
        return absl::InternalError(absl::StrCat(
            "_Arg node with resource output must have non-empty _handle_dtypes "
            "attribute: ",
            node->DebugString()));
      }
      variables->push_back(VariableInput{
          input_edges[i]->src(), input_edges[i]->src_output(), dtypes[0]});
    } else {
      return absl::InternalError(absl::StrCat(
          "Cannot handle variable input with node type other than VarHandleOp "
          "and _Arg: ",
          node->DebugString()));
    }
  }
  return absl::OkStatus();
}

// Builds a NoOp node, used for building control dependencies.
static absl::Status BuildNoopNode(const Node& source, absl::string_view name,
                                  const std::string& device, Graph* graph,
                                  Node** node) {
  NodeDefBuilder builder(name, "NoOp", NodeDebugInfo(source));
  if (!device.empty()) {
    builder.Device(device);
  }
  NodeDef def;
  TF_RETURN_IF_ERROR(builder.Finalize(&def));

  TF_ASSIGN_OR_RETURN(*node, graph->AddNode(def));
  if (!device.empty()) {
    (*node)->set_assigned_device_name(device);
  }
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::ConnectHostComputeNodes(
    Node* compile_node, Node* key_placeholder_node, Graph* graph) {
  // First find all the downstream nodes of the key placeholder node, since we
  // want to delete the connecting edges from key_placeholder_node which would
  // invalidate the out_nodes iterator.
  std::vector<Node*> host_transfer_nodes;
  for (Node* node : key_placeholder_node->out_nodes()) {
    host_transfer_nodes.push_back(node);
  }
  for (Node* node : host_transfer_nodes) {
    int input_index = -1;
    for (int i = 0; i < node->num_inputs(); i++) {
      const Edge* e;
      TF_RETURN_IF_ERROR(node->input_edge(i, &e));
      if (e->src() == key_placeholder_node) {
        if (input_index != -1) {
          return absl::InternalError(absl::StrCat(
              "Node ", node->name(),
              " has multiple input edges from key placeholder node"));
        }
        input_index = e->dst_input();
      }
    }
    if (input_index == -1) {
      return absl::InternalError(
          absl::StrCat("Node ", node->name(),
                       " has no input edge from key placeholder node"));
    }
    const Edge* key_edge;
    TF_RETURN_IF_ERROR(node->input_edge(input_index, &key_edge));
    graph->RemoveEdge(key_edge);
    graph->AddEdge(compile_node, 1, node, input_index);
  }
  graph->RemoveNode(key_placeholder_node);
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::BuildVariableReads(
    absl::Span<const VariableInput> variables, Node* control_predecessor,
    Graph* graph, std::vector<Node*>* variable_reads) {
  variable_reads->resize(variables.size());
  for (int i = 0; i < variables.size(); ++i) {
    std::string name =
        graph->NewName(absl::StrCat(variables[i].node->name(), "/read"));
    NodeDefBuilder builder(name, "ReadVariableOp",
                           NodeDebugInfo(*variables[i].node));

    builder.Attr("dtype", variables[i].dtype);
    builder.Device(variables[i].node->assigned_device_name());
    builder.Input(variables[i].node->name(), 0, DT_RESOURCE);
    NodeDef def;
    TF_RETURN_IF_ERROR(builder.Finalize(&def));

    TF_ASSIGN_OR_RETURN(Node * read_node, graph->AddNode(def));
    (*variable_reads)[i] = read_node;

    read_node->set_requested_device(variables[i].node->requested_device());
    read_node->set_assigned_device_name(
        variables[i].node->assigned_device_name());
    graph->AddEdge(variables[i].node, variables[i].index, read_node, 0);

    graph->AddControlEdge(control_predecessor, read_node);
  }
  return absl::OkStatus();
}

bool DistributedTPURewritePass::ContainsResourceWriteOp(
    const Graph& graph, const FunctionLibraryDefinition& fld) {
  for (const Node* n : graph.nodes()) {
    const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n->type_string());
    if (op_info && op_info->kind() != XlaResourceOpKind::kRead) {
      VLOG(2) << "Found write resource op inside computation";
      return true;
    }
  }
  for (const std::string& func_name : fld.ListFunctionNames()) {
    const FunctionDef* func_def = fld.Find(func_name);
    for (const NodeDef& n : func_def->node_def()) {
      const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n.op());
      if (op_info && op_info->kind() != XlaResourceOpKind::kRead) {
        VLOG(2) << "Found write resource op inside " << func_name;
        return true;
      }
    }
  }
  return false;
}

absl::Status DistributedTPURewritePass::BuildVariableWrites(
    absl::Span<const VariableInput> variables, Node* control_successor,
    absl::Span<const VariableWrite> variable_writes, Graph* graph) {
  CHECK_EQ(variables.size(), variable_writes.size());
  for (int i = 0; i < variables.size(); ++i) {
    const VariableWrite& write = variable_writes[i];
    NodeDebugInfo debug_info(*variables[i].node);

    auto name = [&](std::string suffix) {
      return graph->NewName(
          absl::StrCat(variables[i].node->name(), "/", suffix));
    };

    Node* write_node;
    TF_RETURN_IF_ERROR(
        IncompleteNodeDefBuilder(name("assign"), "AssignVariableOp", debug_info)
            .AddAttr("dtype", variables[i].dtype)
            .Device(variables[i].node->assigned_device_name())
            .Build(graph, &write_node));

    // Colocate the control flow with the variable.
    CondBuilder cb(variables[i].node->name(),
                   variables[i].node->assigned_device_name(), debug_info,
                   graph);

    // Inputs to conditional.
    Node* switch_val;
    TF_RETURN_IF_ERROR(
        cb.AddInput("switch_val", variables[i].dtype,
                    /*device=*/write.value->assigned_device_name(), debug_info,
                    &switch_val));
    Node* switch_var;
    TF_RETURN_IF_ERROR(
        cb.AddInput("switch_var", DT_RESOURCE,
                    /*device=*/variables[i].node->assigned_device_name(),
                    debug_info, &switch_var));
    // Conditionally write the value back.
    graph->AddEdge(variables[i].node, variables[i].index, switch_var, 0);
    graph->AddEdge(switch_var, CondBuilder::kThenBranch, write_node, 0);
    graph->AddEdge(switch_val, CondBuilder::kThenBranch, write_node, 1);
    // Add control edge from the write to value that will be merged. There is no
    // output from the write so this control edge ensures the write completes.
    graph->AddControlEdge(write_node, cb.switch_t());

    graph->AddControlEdge(cb.control_successor(), control_successor);

    graph->AddEdge(write.predicate, write.predicate_output, cb.pred(), 0);
    graph->AddEdge(write.value, write.value_output, switch_val, 0);
  }
  return absl::OkStatus();
}

namespace {

// Computes the shape of the sharded tensor and modifies in place.
absl::Status ComputeShardedArgShapes(TensorShape* shape,
                                     const xla::OpSharding& sharding) {
  if (sharding.type() != xla::OpSharding::OTHER) {
    return absl::OkStatus();
  }
  if (!shape->IsFullyDefined()) {
    return absl::InternalError(
        "Arg shape must be fully defined before sharded shape inference.");
  }
  int sharded_rank = sharding.tile_assignment_dimensions_size();
  if (sharding.replicate_on_last_tile_dim()) {
    sharded_rank--;
  }
  for (int dim_idx = 0; dim_idx < sharded_rank; ++dim_idx) {
    auto sharded_dim = tensorflow::MathUtil::CeilOfRatio<int64_t>(
        shape->dim_size(dim_idx), sharding.tile_assignment_dimensions(dim_idx));
    shape->set_dim(dim_idx, sharded_dim);
  }
  if (sharded_rank != shape->dims()) {
    LOG(WARNING) << "Rank of sharded arg should match sharding spec.  Rank: "
                 << sharded_rank << ", tiled shape: " << shape->DebugString()
                 << ", sharding: " << sharding.DebugString();
  }

  return absl::OkStatus();
}

// Creates nodes for zero-initialized dummy arguments for TPUExecute nodes.
absl::StatusOr<Node*> CreateTpuExecuteDummyArg(
    const TensorShape& var_shape, const DataType& dtype,
    const std::string& host_cpu_device, Node* var_read, int replica_id,
    Graph* graph) {
  absl::Status status;

  // Const - shape_as_tensor
  const std::string name_prefix =
      absl::StrCat(var_read->name(), absl::StrFormat("/dummy_%d", replica_id));
  NodeDef shape_tensor_def;
  shape_tensor_def.set_op("Const");
  shape_tensor_def.set_name(graph->NewName(
      absl::StrCat(name_prefix, "/Initializer/zeros/shape_as_tensor")));
  shape_tensor_def.set_device(host_cpu_device);
  AddNodeAttr("dtype", DT_INT32, &shape_tensor_def);
  TensorProto tensorshape_proto;
  tensorshape_proto.set_dtype(DT_INT32);
  for (int i = 0; i < var_shape.dims(); ++i) {
    tensorshape_proto.add_int_val(var_shape.dim_size(i));
  }
  TensorShape shape_shape({var_shape.dims()});
  shape_shape.AsProto(tensorshape_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensorshape_proto, &shape_tensor_def);
  TF_ASSIGN_OR_RETURN(Node * shape_as_tensor_node,
                      graph->AddNode(shape_tensor_def));

  // Const - initializer value
  NodeDef init_val_def;
  init_val_def.set_op("Const");
  init_val_def.set_name(graph->NewName(
      absl::StrCat(name_prefix, "/Initializer/zeros/const_val")));
  init_val_def.set_device(host_cpu_device);
  TensorProto tensor_proto;
  tensor_proto.set_dtype(dtype);
  const absl::flat_hash_set<DataType> kSupportedIntTypes = {
      DT_INT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8, DT_QINT8, DT_QUINT8};
  if (dtype == DT_FLOAT) {
    tensor_proto.add_float_val(0.0f);
  } else if (kSupportedIntTypes.contains(dtype)) {
    tensor_proto.add_int_val(0);
  } else if (dtype == DT_BFLOAT16 || dtype == DT_HALF) {
    tensor_proto.add_half_val(0);
  } else if (dtype == DT_UINT32) {
    tensor_proto.add_uint32_val(0);
  } else if (dtype == DT_BOOL) {
    tensor_proto.add_bool_val(false);
  } else {
    return absl::InternalError(absl::StrCat(
        "Unable to create zero-init dummy arg tensor for variable ",
        var_read->name(), " of type ", dtype));
  }
  TensorShape scalar_shape({});
  scalar_shape.AsProto(tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensor_proto, &init_val_def);
  AddNodeAttr("dtype", dtype, &init_val_def);
  TF_ASSIGN_OR_RETURN(Node * init_val_node, graph->AddNode(init_val_def));

  // Fill node
  NodeDef fill_def;
  fill_def.set_op("Fill");
  fill_def.set_device(host_cpu_device);
  fill_def.set_name(
      graph->NewName(absl::StrCat(name_prefix, "/Initializer/zeros")));
  AddNodeAttr("T", dtype, &fill_def);
  AddNodeAttr("index_type", DT_INT32, &fill_def);
  TF_ASSIGN_OR_RETURN(Node * fill_node, graph->AddNode(fill_def));
  graph->AddEdge(shape_as_tensor_node, 0, fill_node, 0);
  graph->AddEdge(init_val_node, 0, fill_node, 1);

  return fill_node;
}

// Creates dummy inputs for partitioned variables that are using XLA broadcast
// for inputs.
absl::Status CreatePartitionedDummyVarArgs(
    const xla::OpSharding& sharding, const int num_replicas,
    const int replica_id, const InferredShape& raw_shape, Node* orig_var_read,
    const int orig_arg_num, DataType dtype, const std::string& device,
    Graph* graph, const std::vector<std::vector<std::string>>& tpu_device_names,
    absl::btree_map<ShardedPerHostInputIndex, Node*>* per_host_index,
    std::map<ShardedInputIndex, ShardedInputInfo>*
        arg_index_to_sharded_input_map) {
  ShardedInputIndex input_index{replica_id, orig_arg_num};
  auto iter = arg_index_to_sharded_input_map->find(input_index);
  if (iter != arg_index_to_sharded_input_map->end()) {
    return absl::OkStatus();
  }
  const int repeat = sharding.replicate_on_last_tile_dim()
                         ? *sharding.tile_assignment_dimensions().rbegin()
                         : 1;
  const int num_shards = sharding.tile_assignment_devices_size() / repeat;

  TensorShape var_shape;
  if (!raw_shape.handle_shape.AsTensorShape(&var_shape) &&
      !raw_shape.shape.AsTensorShape(&var_shape)) {
    return absl::FailedPreconditionError("Failed to read arg shape.");
  }
  TF_RETURN_IF_ERROR(ComputeShardedArgShapes(&var_shape, sharding));

  for (int replica = 1; replica < num_replicas; ++replica) {
    std::vector<NodeOut> sharded_inputs_list(
        sharding.tile_assignment_devices_size());
    for (int i = 0; i < num_shards; ++i) {
      for (int j = 0; j < repeat; ++j) {
        const int index = i * repeat + j;
        const int core = sharding.tile_assignment_devices(index);
        std::string host_device;
        TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
            tpu_device_names[replica][core], &host_device));
        ShardedPerHostInputIndex idx{host_device, orig_arg_num};
        if (!per_host_index->contains(idx)) {
          TF_ASSIGN_OR_RETURN(
              auto dummy_node,
              CreateTpuExecuteDummyArg(var_shape, dtype, host_device,
                                       orig_var_read, replica, graph));
          (*per_host_index)[idx] = dummy_node;
        }
        sharded_inputs_list[core] = {(*per_host_index)[idx], /*index=*/0};
      }
    }
    ShardedInputInfo sharded_input_info{nullptr,
                                        std::move(sharded_inputs_list)};
    (*arg_index_to_sharded_input_map)[{replica, orig_arg_num}] =
        sharded_input_info;
  }

  return absl::OkStatus();
}

// Helper that creates an IdentityN node containing all of the variables
// values on CPU device 'device', except for those that will be split across
// cores. (For split variables, this may cause additional cross-host data
// transfers if more than 1 devices share the same variable partition on a
// remote host.)
//
// A previous iteration of this code built one Identity node per TPU core per
// variable, but this can rapidly become hundreds of thousands of nodes. This
// formulation creates a single IdentityN node containing all of the variables
// on each host. This may cause some unnecessary variable copies if only a
// subset of hosts consume a given variable, but has the virtue of being
// simple, and most models use pure replication where all cores want all the
// variables.
//
// If enable_xla_param_broadcast is set to true, then per-host dummy
// tensor args are created on all hosts except for the primary host. In this
// scheme, the dummy args feed the IdentityN node on their local host. All
// are zero-initialized.
//
// Returns the node and its output index to be consumed by TPUExecute for the
// requested variable index.
absl::StatusOr<NodeOut> CreateOrGetPerHostVariableCopy(
    const std::string& host_cpu_device, int64_t var_index,
    const std::vector<Node*>& variable_reads,
    const DistributedTPURewritePass::ParameterInfo& params_info,
    const std::vector<xla::OpSharding>& arg_shardings,
    const Node& replicate_node, const bool enable_xla_param_broadcast,
    const bool mpmd, const int num_cores_per_replica, int replica_id,
    const std::vector<InferredShape>& arg_shapes,
    absl::flat_hash_map<std::string, std::vector<NodeOut>>* per_host_var_copies,
    Graph* graph) {
  auto it = per_host_var_copies->find(host_cpu_device);
  if (it != per_host_var_copies->end()) {
    return it->second[var_index];
  }

  DataTypeVector dtypes;
  // Per-variable data source for TPUExecute.
  std::vector<NodeOut> index_mapping;
  index_mapping.reserve(variable_reads.size());
  dtypes.reserve(variable_reads.size());
  for (int64_t i = 0; i < variable_reads.size(); ++i) {
    Node* read = variable_reads[i];
    int64_t orig_arg_num = i + params_info.NumPerReplicaArgs() +
                           params_info.NumDistributedArgs() +
                           params_info.NumBroadcastArgs();
    if (arg_shardings[orig_arg_num].type() != xla::OpSharding::OTHER) {
      // We haven't built the IdentityN node yet, so temporarily use nullptr.
      index_mapping.push_back(
          NodeOut{nullptr, static_cast<int>(dtypes.size())});
      dtypes.push_back(read->output_type(0));
    } else {
      // Do not copy the full tensor of partitioned variables.
      index_mapping.push_back(NodeOut{read, 0});
    }
  }
  NodeDef ndef;
  ndef.set_name(graph->NewName(
      absl::StrCat(replicate_node.name(), "/", kTpuExecuteStagingNodeName)));
  ndef.set_op(kTpuExecuteStagingOp);
  ndef.set_device(host_cpu_device);
  AddNodeAttr("T", dtypes, &ndef);
  // TF meta-optimizer should skip this node for constant folding.
  AddNodeAttr("_tpu_avoid_constant_fold", "not_used", &ndef);
  TF_ASSIGN_OR_RETURN(Node * id_node, graph->AddNode(ndef));
  id_node->set_assigned_device_name(host_cpu_device);

  for (int64_t i = 0; i < variable_reads.size(); ++i) {
    Node* read = variable_reads[i];
    int64_t orig_arg_num = i + params_info.NumPerReplicaArgs() +
                           params_info.NumDistributedArgs() +
                           params_info.NumBroadcastArgs();
    DataType dtype = read->output_type(0);
    const bool use_xla_broadcast =
        EnableXlaParamBroadcast(enable_xla_param_broadcast, mpmd, params_info,
                                orig_arg_num, dtype) &&
        replica_id != 0;
    if (index_mapping[i].node == nullptr) {
      // Fill index_mapping with the actual IdentityN node.
      index_mapping[i].node = id_node;
      if (!use_xla_broadcast) {
        // Add the variable read edge to id_node.
        graph->AddEdge(variable_reads[i], 0, id_node, index_mapping[i].index);
      } else {
        // XLA param broadcast mode is enabled.  Create zero-valued dummy
        // tensors to use as variable args in the TPUExecuteOp, instead of
        // original variable reads.
        TensorShape var_shape;
        auto inferred_shape = arg_shapes[orig_arg_num];
        if (!inferred_shape.handle_shape.AsTensorShape(&var_shape) &&
            !inferred_shape.shape.AsTensorShape(&var_shape)) {
          return absl::FailedPreconditionError("Failed to read arg shape.");
        }
        TF_ASSIGN_OR_RETURN(
            Node * dummy_read,
            CreateTpuExecuteDummyArg(var_shape, dtype, host_cpu_device,
                                     variable_reads[i], replica_id, graph));
        graph->AddEdge(dummy_read, 0, id_node, index_mapping[i].index);
      }
    }
  }

  auto result = index_mapping[var_index];
  (*per_host_var_copies)[host_cpu_device] = std::move(index_mapping);
  return result;
}

}  // namespace

absl::Status DistributedTPURewritePass::BuildExecuteNodes(
    const ParameterInfo& params_info, int num_tasks, int num_cores_per_replica,
    const Node& replicate_node, const std::vector<std::string>& arg_names,
    const DataTypeVector& arg_types,
    const std::vector<InferredShape>& arg_shapes,
    const DataTypeVector& retval_types,
    const std::vector<xla::OpSharding>& arg_shardings,
    const std::vector<xla::OpSharding>& retval_shardings,
    const std::vector<std::vector<std::string>>& tpu_device_names,
    Node* compile_node, const std::vector<Node*>& variable_reads,
    Node* control_predecessor, Node* control_successor, Node* multilock_acquire,
    std::vector<VariableWrite>* variable_writes, Graph* graph) {
  VLOG(1) << "BuildExecuteNodes " << replicate_node.DebugString();
  TF_RET_CHECK(params_info.NumReplicas() == tpu_device_names.size());

  const int num_variables = variable_reads.size();
  const int num_retvals_per_replica = retval_types.size();

  variable_writes->resize(num_variables);

  std::vector<const Edge*> replicate_input_edges;
  TF_RETURN_IF_ERROR(replicate_node.input_edges(&replicate_input_edges));

  // Map from replicate input index to the fan_in node;
  absl::flat_hash_map<int, std::vector<NodeAndPort>>
      replicate_input_fan_in_nodes;
  absl::flat_hash_map<int, std::vector<Node*>> replicate_output_fan_out_nodes;
  absl::flat_hash_map<int, std::vector<int>>
      replicate_output_fan_out_dst_inputs;
  std::vector<Node*> to_be_removed_nodes;

  const bool use_spmd =
      UseSpmdForXlaPartitioning(&replicate_node) && allow_xla_spmd_partition_;
  const bool mpmd = (num_cores_per_replica > 1) && !use_spmd;

  for (const Edge* e : replicate_input_edges) {
    if (_IsTPUPartitionedInput(e->src())) {
      int num_users = 0;
      for (const auto& ue : e->src()->out_edges()) {
        if (!ue->IsControlEdge()) ++num_users;
      }
      if (num_users != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            e->src()->name(), " must only have one user. Found ", num_users));
      }
      to_be_removed_nodes.push_back(e->src());
      std::vector<NodeAndPort>& nodes =
          replicate_input_fan_in_nodes[e->dst_input()];
      nodes.resize(num_cores_per_replica, NodeAndPort(nullptr, 0));
      VLOG(2) << "allocate " << num_cores_per_replica
              << " for replicate_input_fan_in_nodes[" << e->dst_input() << "]";

      std::vector<const Edge*> fan_in_edges;
      TF_RETURN_IF_ERROR(e->src()->input_edges(&fan_in_edges));

      bool is_packed = false;
      TF_RET_CHECK((e->src()->type_string() == kTPUPartitionedInput) ||
                   TryGetNodeAttr(e->src()->def(), "is_packed", &is_packed));

      int num_fan_in_edges = fan_in_edges.size();
      TF_RET_CHECK(is_packed || (num_fan_in_edges == num_cores_per_replica));

      for (int i = 0; i < num_cores_per_replica; ++i) {
        const Edge* fe = fan_in_edges[i % num_fan_in_edges];
        nodes[i].node = fe->src();
        nodes[i].port = fe->src_output();
        VLOG(2) << "replicate_input_fan_in_nodes[" << e->dst_input() << "]["
                << i << "] = " << fe->src()->name();
      }
    }
  }

  // Replicate output edges are sorted by replica id and then by outputs for
  // each replica. For example, if TPU Computation has outputs (output_1,
  // output_2, and output_3) and number of replicas is 2, then
  // replicate_output_edges order would be:
  // output_1_replica_1, output_2_replica_1, output_3_replica_1,
  // output_1_replica_2, output_2_replica_2, output_3_replica_2.
  std::vector<const Edge*> replicate_output_edges(replicate_node.num_outputs(),
                                                  nullptr);
  for (const Edge* edge : replicate_node.out_edges()) {
    if (edge->IsControlEdge()) continue;

    int num_partitioned_outputs = 0;

    for (const Edge* out_edge : edge->dst()->out_edges()) {
      if (_IsTPUPartitionedOutput(out_edge->dst())) {
        num_partitioned_outputs++;
        // Paths between replicate_node and replicate_output_fan_out_nodes:
        // ReplicateNode->TpuOutIdenity->kTPUPartitionedOutput->fan-out-nodes
        TF_RET_CHECK(edge->dst()->out_edges().size() == 1);
        to_be_removed_nodes.push_back(edge->dst());
        to_be_removed_nodes.push_back(out_edge->dst());
        // Get the right replicated id from the replicate_output_edge.
        std::vector<Node*>& nodes =
            replicate_output_fan_out_nodes[edge->src_output()];
        std::vector<int>& dst_inputs =
            replicate_output_fan_out_dst_inputs[edge->src_output()];
        nodes.resize(num_cores_per_replica, nullptr);
        dst_inputs.resize(num_cores_per_replica, 0);
        TF_RET_CHECK(out_edge->dst()->out_edges().size() ==
                     num_cores_per_replica);

        for (const Edge* fe : out_edge->dst()->out_edges()) {
          nodes[fe->src_output()] = fe->dst();
          dst_inputs[fe->src_output()] = fe->dst_input();
          VLOG(2) << "replicate_output_fan_out_nodes[" << out_edge->src_output()
                  << "][" << fe->src_output()
                  << "] = " << fe->dst()->DebugString() << " with dst_input "
                  << fe->dst_input();
        }
      }
    }
    replicate_output_edges[edge->src_output()] = edge;
    if (num_partitioned_outputs > 1) {
      return absl::InvalidArgumentError(
          "More than one TPUPartitionedOutput per replicated output.");
    }
  }

  const int num_execute_args =
      arg_shardings.size() - params_info.NumGuaranteedConstants();
  // Inverts the arg_shardings and retval_shardings mappings to
  // form core -> {argument number} maps.
  std::vector<std::vector<int>> core_arg_nums(num_cores_per_replica);
  for (int i = 0; i < num_execute_args; ++i) {
    const auto& sharding = arg_shardings[i];
    if (sharding.type() == xla::OpSharding::MAXIMAL) {
      int core = sharding.tile_assignment_devices(0);
      TF_RETURN_IF_ERROR(ValidateCoreNumber(core, num_cores_per_replica));
      core_arg_nums[core].push_back(i);
    } else if (sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t core : sharding.tile_assignment_devices()) {
        core_arg_nums[core].push_back(i);
      }
    } else if (sharding.type() == xla::OpSharding::REPLICATED) {
      for (int core = 0; core < num_cores_per_replica; ++core) {
        core_arg_nums[core].push_back(i);
      }
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported argument sharding for arg=", arg_names[i], " shape=",
          arg_shapes[i].shape.DebugString(), ": ", sharding.DebugString()));
    }
  }
  std::vector<std::vector<int>> core_retval_nums(num_cores_per_replica);
  for (int i = 0; i < retval_shardings.size(); ++i) {
    const auto& sharding = retval_shardings[i];
    if (sharding.type() == xla::OpSharding::MAXIMAL) {
      int core = sharding.tile_assignment_devices(0);
      TF_RETURN_IF_ERROR(ValidateCoreNumber(core, num_cores_per_replica));
      core_retval_nums[core].push_back(i);
    } else if (sharding.type() == xla::OpSharding::REPLICATED) {
      for (int core = 0; core < num_cores_per_replica; ++core) {
        core_retval_nums[core].push_back(i);
      }
    } else if (sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t core : sharding.tile_assignment_devices()) {
        core_retval_nums[core].push_back(i);
      }
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported argument sharding: ", sharding.DebugString()));
    }
  }

  // Maps host device name to a list of per-variable pairs (variable_copy_node,
  // output_index_of_copy_node).
  absl::flat_hash_map<std::string, std::vector<NodeOut>> per_host_var_copies;

  Node* execute_successor = control_successor;

  int num_total_cores = params_info.NumReplicas() * num_cores_per_replica;
  if (enable_multicore_locking_ && num_total_cores > 1) {
    // Add a node to release exclusive access once all the cores have finished
    // execution.
    NodeDef lock_def;
    lock_def.set_name(graph->NewName(
        absl::StrCat(compile_node->name(), "/", "tpu_release_multilock")));
    lock_def.set_op("ConsumeTpuMultilock");
    MergeDebugInfo(NodeDebugInfo(replicate_node.def()), &lock_def);
    TF_ASSIGN_OR_RETURN(Node * multilock_release, graph->AddNode(lock_def));
    multilock_release->set_assigned_device_name(
        compile_node->assigned_device_name());
    TF_RET_CHECK(multilock_acquire != nullptr);
    graph->AddEdge(multilock_acquire, 0, multilock_release, 0);
    graph->AddControlEdge(multilock_release, control_successor);
    // Make sure all execute Ops happen before the multilock_release.
    execute_successor = multilock_release;
  }

  // Mapping from original resource arg number to a second level map. Second
  // level map is from core id to output index of updated variable value.
  absl::flat_hash_map<int, absl::flat_hash_map<int, int>>
      orig_arg_num_to_output_index_mapping;
  // Mapping from retval index to a second level map. Second level map is from
  // core id to output index of sharded output value.
  std::unordered_map<int, absl::flat_hash_map<int, int>>
      retval_index_to_output_index_mapping;

  // Represents mapping of argument index of sharded input to each
  // TPUExecute node to its corresponding Split node and its output index
  // from which sharded input will be fed into TPUExecute node.
  std::map<ShardedInputIndex, ShardedInputInfo> input_index_to_sharded_inputs;

  // Additional map of {host, arg_num} to dummy input. Per-task copies of the
  // inputs reduces cross-task communication and allows sharing across replicas.
  absl::btree_map<ShardedPerHostInputIndex, Node*> sharded_per_host_index;

  // Builds one TPUExecute node per core per replica.
  std::vector<std::vector<Node*>> execute_nodes(params_info.NumReplicas());
  for (int core = 0; core < num_cores_per_replica; ++core) {
    DataTypeVector core_retval_types;
    for (int output : core_retval_nums[core]) {
      core_retval_types.push_back(retval_types[output]);
    }
    DataTypeVector core_arg_types;
    std::vector<int> core_variable_writes;
    for (int input : core_arg_nums[core]) {
      // Resource variables can be passed either by reference (as a DT_RESOURCE)
      // tensor or by value (as the variable's current value). Per-replica or
      // distributed resource arguments are always passed by reference and
      // broadcast variables are always passed by value.
      if (arg_types[input] == DT_RESOURCE &&
          !params_info.IsPerReplicaArg(input) &&
          !params_info.IsDistributedArg(input)) {
        DataType handle_type = arg_shapes[input].handle_type;
        TF_RET_CHECK(handle_type != DT_INVALID) << DataTypeString(handle_type);
        core_arg_types.push_back(handle_type);
        int base = input - params_info.NumPerReplicaArgs() -
                   params_info.NumDistributedArgs() -
                   params_info.NumBroadcastArgs();
        // Variables passed by value will have a corresponding additional output
        // containing an updated value for the variable.
        core_variable_writes.push_back(base);
        core_retval_types.push_back(handle_type);
      } else {
        core_arg_types.push_back(arg_types[input]);
      }
    }

    NodeDef def;
    def.set_op("TPUExecute");
    MergeDebugInfo(NodeDebugInfo(replicate_node.def()), &def);
    AddNodeAttr("Targs", core_arg_types, &def);
    AddNodeAttr("Tresults", core_retval_types, &def);

    // If the producer name was set during inference, propagate the information
    // to the TPUExecute op so it can be accessed during metric collection.
    std::string producer_name;
    absl::Status status =
        GetNodeAttr(replicate_node.attrs(), "_producer_name", &producer_name);
    if (status.ok()) {
      AddNodeAttr("_producer_name", producer_name, &def);
    }

    for (int64_t replica = 0; replica < params_info.NumReplicas(); ++replica) {
      def.set_name(absl::StrCat(replicate_node.name(), "/_execute_", replica,
                                "_", core));

      TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(def));
      execute_nodes[replica].push_back(node);

      node->set_assigned_device_name(tpu_device_names[replica][core]);

      // Add control edges to ensure that execution happens after
      // `control_predecessor`, happens before `execute_successor`, and is
      // triggered by evaluating any operator that depends on the original
      // TPUReplicate operator. See the comment at the top of the header file
      // for more details.
      graph->AddControlEdge(control_predecessor, node);
      graph->AddControlEdge(node, execute_successor);

      // Add data input edges.
      for (int64_t i = 0; i < core_arg_nums[core].size(); ++i) {
        int64_t orig_arg_num = core_arg_nums[core][i];
        VLOG(2) << " replica " << replica << " core " << core << " i " << i
                << " orig_arg_num " << orig_arg_num;
        const bool is_per_replica_arg =
            params_info.IsPerReplicaArg(orig_arg_num);
        if (is_per_replica_arg || params_info.IsDistributedArg(orig_arg_num)) {
          // Per-replica input and distributed input
          const int64_t input_num =
              is_per_replica_arg ? replica * params_info.NumPerReplicaArgs() +
                                       core_arg_nums[core][i]
                                 : params_info.NumReplicas() *
                                           params_info.NumPerReplicaArgs() +
                                       core_arg_nums[core][i] -
                                       params_info.NumPerReplicaArgs();

          const Edge* edge = replicate_input_edges[input_num];
          VLOG(2) << "replicate_input_edges[" << input_num << "]";
          DataType dtype = edge->src()->output_type(edge->src_output());
          if (dtype == DT_RESOURCE) {
            DataType handle_dtype = arg_shapes[orig_arg_num].handle_type;
            if (std::find(kTpuAllTypes.begin(), kTpuAllTypes.end(),
                          handle_dtype) == kTpuAllTypes.end()) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported resource variable data type for TPU: ",
                  DataTypeString(handle_dtype), ", caused by output ",
                  edge->src()->name(), ":", edge->src_output()));
            }
          } else {
            if (std::find(kTpuAllTypes.begin(), kTpuAllTypes.end(), dtype) ==
                kTpuAllTypes.end()) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported data type for TPU: ", DataTypeString(dtype),
                  ", caused by output ", edge->src()->name(), ":",
                  edge->src_output()));
            }
          }
          if (IsSplitSharding(arg_shardings[orig_arg_num])) {
            // Don't automatically add a split node when input node is
            // kTPUPartitionedInput
            if (_IsTPUPartitionedInput(edge->src())) {
              VLOG(2)
                  << "Connect "
                  << replicate_input_fan_in_nodes[input_num][core].node->name()
                  << " to " << node->name() << " at " << i;
              graph->AddEdge(replicate_input_fan_in_nodes[input_num][core].node,
                             replicate_input_fan_in_nodes[input_num][core].port,
                             node, i);
            } else {
              if (dtype == DT_RESOURCE) {
                return absl::InvalidArgumentError(absl::StrCat(
                    "Tiled sharding for per-replica DT_RESOURCE input must",
                    "be TPUPartitionedInput. Here got ",
                    edge->src()->type_string()));
              }
              const xla::OpSharding& sharding = arg_shardings[orig_arg_num];

              ShardedInputInfo sharded_input_info;
              if (use_nd_sharding_ops_ && is_per_replica_arg) {
                TF_ASSIGN_OR_RETURN(
                    sharded_input_info,
                    CreateOrGetXlaSplitNodeForShardedPerReplicaArg(
                        sharding, replica, orig_arg_num, dtype,
                        PartialTensorShape(), edge->src(), edge->src_output(),
                        graph, &input_index_to_sharded_inputs));
              } else if (use_nd_sharding_ops_) {
                TF_ASSIGN_OR_RETURN(
                    sharded_input_info,
                    CreateOrGetXlaSplitNodeForDistributedArg(
                        sharding, params_info.NumReplicas(), replica,
                        orig_arg_num, dtype, PartialTensorShape(), edge->src(),
                        edge->src_output(), graph,
                        &input_index_to_sharded_inputs));
              } else {
                TF_ASSIGN_OR_RETURN(
                    sharded_input_info,
                    CreateOrGetSplitNodesForInputSharding(
                        sharding, orig_arg_num, dtype, PartialTensorShape(),
                        replica, edge->src_output(), edge->src(),
                        control_predecessor, graph,
                        &input_index_to_sharded_inputs));
              }

              NodeOut split_node_and_index =
                  sharded_input_info.sharded_inputs.at(core);
              // Connect with Split node output.
              graph->AddEdge(split_node_and_index.node,
                             split_node_and_index.index, node, i);
            }
          } else if (_IsTPUPartitionedInput(edge->src()) &&
                     IsReplicatedSharding(arg_shardings[orig_arg_num])) {
            graph->AddEdge(replicate_input_fan_in_nodes[input_num][core].node,
                           replicate_input_fan_in_nodes[input_num][core].port,
                           node, i);
          } else {
            graph->AddEdge(edge->src(), edge->src_output(), node, i);
          }
        } else if (params_info.IsBroadcastArg(orig_arg_num)) {
          // Broadcast input.
          int64_t input_num = params_info.FirstBroadcastArgFromHost() +
                              core_arg_nums[core][i] -
                              params_info.NumPerReplicaArgs() -
                              params_info.NumDistributedArgs();
          const Edge* edge = replicate_input_edges[input_num];
          DataType dtype = edge->src()->output_type(edge->src_output());
          if (std::find(kTpuAllTypes.begin(), kTpuAllTypes.end(), dtype) ==
              kTpuAllTypes.end()) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Unsupported data type for TPU: ", DataTypeString(dtype),
                ", caused by output ", edge->src()->name(), ":",
                edge->src_output()));
          }
          graph->AddEdge(edge->src(), edge->src_output(), node, i);
        } else {
          // Variable input.
          int64_t variable_num =
              orig_arg_num - params_info.NumPerReplicaArgs() -
              params_info.NumDistributedArgs() - params_info.NumBroadcastArgs();
          TF_RET_CHECK(variable_num < num_variables);

          Node* variable_read = variable_reads[variable_num];
          DataType dtype = variable_read->output_type(0);
          if (std::find(kTpuAllTypes.begin(), kTpuAllTypes.end(), dtype) ==
              kTpuAllTypes.end()) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Unsupported resource variable data type for TPU: ",
                DataTypeString(dtype), ", caused by ReadVariableOp ",
                variable_read->DebugString()));
          }
          DeviceNameUtils::ParsedName requested_device;
          std::string requested = variable_read->requested_device();
          TF_RET_CHECK(
              DeviceNameUtils::ParseFullName(requested, &requested_device));
          if (requested_device.type != "TPU") {
            // Stage the value via the CPU device on the remote host. The graph
            // partitioner will introduce an intermediate copy rather than
            // copying the same tensor multiple times across the network, and we
            // would prefer that intermediate copy to be in host memory to avoid
            // running out of memory if the TPUExecute op on the staging device
            // starts running before the _Send ops to the other TPU devices on
            // the same host complete. We don't do this if the variables are
            // already placed on TPU, otherwise it will cause an unnecessary
            // round trip copy.
            // TODO(b/79580121): give each replica its own on-device variable
            // replica and then delete this code.
            std::string device;
            TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
                tpu_device_names[replica][core], &device));
            TF_ASSIGN_OR_RETURN(
                auto var_data,
                CreateOrGetPerHostVariableCopy(
                    device, variable_num, variable_reads, params_info,
                    arg_shardings, replicate_node, enable_xla_param_broadcast_,
                    mpmd, num_cores_per_replica, replica, arg_shapes,
                    &per_host_var_copies, graph));

            if (IsSplitSharding(arg_shardings[orig_arg_num])) {
              ShardedInputInfo sharded_input_info;

              if (EnableXlaParamBroadcast(enable_xla_param_broadcast_, mpmd,
                                          params_info, orig_arg_num, dtype)) {
                // Populates the sharded dummy vars for non-zero replicas.
                TF_RETURN_IF_ERROR(CreatePartitionedDummyVarArgs(
                    arg_shardings[orig_arg_num], params_info.NumReplicas(),
                    replica, arg_shapes[orig_arg_num], var_data.node,
                    orig_arg_num, dtype, device, graph, tpu_device_names,
                    &sharded_per_host_index, &input_index_to_sharded_inputs));
              }

              if (use_nd_sharding_ops_) {
                TF_ASSIGN_OR_RETURN(
                    sharded_input_info,
                    CreateOrGetXlaSplitNodeForVariableArg(
                        arg_shardings[orig_arg_num], params_info.NumReplicas(),
                        replica, orig_arg_num,
                        arg_shapes[orig_arg_num].handle_type,
                        arg_shapes[orig_arg_num].handle_shape, var_data.node,
                        var_data.index, graph, &to_be_removed_nodes,
                        &input_index_to_sharded_inputs));
              } else {
                TF_ASSIGN_OR_RETURN(
                    sharded_input_info,
                    CreateOrGetSplitNodesForInputSharding(
                        arg_shardings[orig_arg_num], orig_arg_num,
                        arg_shapes[orig_arg_num].handle_type,
                        arg_shapes[orig_arg_num].handle_shape, replica,
                        var_data.index, var_data.node, control_predecessor,
                        graph, &input_index_to_sharded_inputs));
              }

              NodeOut split_node_and_index =
                  sharded_input_info.sharded_inputs[core];
              // Connect with Split node output.
              graph->AddEdge(split_node_and_index.node,
                             split_node_and_index.index, node, i);

            } else {
              graph->AddEdge(var_data.node, var_data.index, node, i);
            }
          } else {
            graph->AddEdge(variable_reads[variable_num], 0, node, i);
          }
        }
      }

      // Adds a program input edge from the compiler.
      graph->AddEdge(compile_node, core + 1, node, node->num_inputs() - 1);

      // Add data output edges.
      int num_outputs = core_retval_nums[core].size();
      for (int i = 0; i < num_outputs; ++i) {
        int output_num =
            replica * num_retvals_per_replica + core_retval_nums[core][i];
        const auto& sharding = retval_shardings[core_retval_nums[core][i]];
        if (IsSplitSharding(sharding)) {
          int retval_index = core_retval_nums[core][i];
          retval_index_to_output_index_mapping[retval_index][core] = i;
          bool is_last_core =
              core ==
              *std::max_element(sharding.tile_assignment_devices().begin(),
                                sharding.tile_assignment_devices().end());
          bool isPartitionOutNode = false;

          const Edge* e = replicate_output_edges[output_num];
          const Edge* e_out;
          for (const Edge* out_edge : e->dst()->out_edges()) {
            if (_IsTPUPartitionedOutput(out_edge->dst())) {
              isPartitionOutNode = true;
              e_out = out_edge;
            }
          }
          if (isPartitionOutNode) {
            graph->AddEdge(
                node, i, replicate_output_fan_out_nodes[output_num][core],
                replicate_output_fan_out_dst_inputs[output_num][core]);
            VLOG(2) << "Connect " << node->name() << " at " << i << " to "
                    << replicate_output_fan_out_nodes[output_num][core]->name()
                    << " at "
                    << replicate_output_fan_out_dst_inputs[output_num][core];
            if (is_last_core) {
              graph->RemoveEdge(e);
              graph->RemoveEdge(e_out);
            }
            continue;
          }

          // Do this in the iteration of last core in tile assignment, so all
          // TPUExecute nodes have been created.
          if (!is_last_core) {
            continue;
          }

          // Add a Concat node.
          std::vector<NodeOut> orig_inputs;
          for (int64_t tile_index = 0;
               tile_index < sharding.tile_assignment_devices_size();
               ++tile_index) {
            int64_t last_tile_dim_size =
                *sharding.tile_assignment_dimensions().rbegin();
            if (sharding.replicate_on_last_tile_dim() &&
                tile_index % last_tile_dim_size != 0) {
              continue;
            }
            int64_t core_id = sharding.tile_assignment_devices(tile_index);
            int core_retval_index =
                retval_index_to_output_index_mapping[retval_index][core_id];
            orig_inputs.push_back(
                NodeOut{execute_nodes[replica][core_id],
                        static_cast<int>(
                            core_retval_nums[core_id][core_retval_index])});
          }
          DataType dtype = e->src()->output_type(e->src_output());
          Node* concat_node = nullptr;
          if (use_nd_sharding_ops_) {
            TF_ASSIGN_OR_RETURN(
                concat_node, CreateXlaConcatNode(
                                 sharding, replica, dtype,
                                 /*partial_tensor_shape=*/PartialTensorShape(),
                                 orig_inputs, /*device=*/"", graph));
          } else {
            TF_ASSIGN_OR_RETURN(
                concat_node,
                CreateConcatNodesForRetval(
                    sharding, dtype, /*inferred_shape=*/PartialTensorShape(),
                    replica, orig_inputs, graph, /*device=*/""));
          }

          const Edge* edge = replicate_output_edges[output_num];
          Node* dst = edge->dst();
          int dst_input = edge->dst_input();
          graph->RemoveEdge(edge);
          graph->AddEdge(concat_node, 0, dst, dst_input);

          continue;
        }

        // If this is a replicated output, outputs on all cores will be the
        // same, and we only take the output from core 0.
        if (IsReplicatedSharding(sharding) && core != 0) {
          continue;
        }

        // If output has maximal sharding, make sure we only use output from
        // TPUExecute node with logical core id equal to core id defined by the
        // xla sharding.
        if (sharding.type() == xla::OpSharding::MAXIMAL &&
            core != sharding.tile_assignment_devices(0)) {
          continue;
        }

        const Edge* replicate_edge_to_replace =
            replicate_output_edges[output_num];
        Node* dst = replicate_edge_to_replace->dst();
        int dst_input = replicate_edge_to_replace->dst_input();
        graph->RemoveEdge(replicate_edge_to_replace);
        graph->AddEdge(node, i, dst, dst_input);
      }

      // Feed the updated variable values from the first replica to the
      // variable write nodes.
      if (replica == 0) {
        for (int i = 0; i < core_variable_writes.size(); ++i) {
          int orig_arg_num =
              core_variable_writes[i] + params_info.NumPerReplicaArgs() +
              params_info.NumDistributedArgs() + params_info.NumBroadcastArgs();
          const auto& sharding = arg_shardings[orig_arg_num];
          // If this is a tiling sharded variable, concat variable updates from
          // all cores.
          if (IsSplitSharding(sharding)) {
            orig_arg_num_to_output_index_mapping[orig_arg_num][core] = i;

            // Do this in the iteration of last core in tile assignment, so all
            // TPUExecute nodes have been created.
            if (core !=
                *std::max_element(sharding.tile_assignment_devices().begin(),
                                  sharding.tile_assignment_devices().end())) {
              continue;
            }

            // Add a Concat node.
            std::vector<NodeOut> orig_inputs;
            for (int64_t tile_index = 0;
                 tile_index < sharding.tile_assignment_devices_size();
                 ++tile_index) {
              int64_t last_tile_dim_size =
                  *sharding.tile_assignment_dimensions().rbegin();
              if (sharding.replicate_on_last_tile_dim() &&
                  tile_index % last_tile_dim_size != 0) {
                continue;
              }
              int64_t core_id = sharding.tile_assignment_devices(tile_index);
              int core_retval_num =
                  orig_arg_num_to_output_index_mapping[orig_arg_num][core_id];
              orig_inputs.push_back(
                  NodeOut{execute_nodes[0][core_id],
                          static_cast<int>(core_retval_nums[core_id].size() +
                                           core_retval_num)});
            }

            // Use the variable read's device for the concat. They should both
            // be collocated with the variable.
            absl::string_view device =
                variable_reads[core_variable_writes[i]]->assigned_device_name();
            Node* concat_node = nullptr;
            if (use_nd_sharding_ops_) {
              TF_ASSIGN_OR_RETURN(
                  concat_node,
                  CreateXlaConcatNode(sharding, replica,
                                      arg_shapes[orig_arg_num].handle_type,
                                      arg_shapes[orig_arg_num].handle_shape,
                                      orig_inputs, device, graph));
            } else {
              TF_ASSIGN_OR_RETURN(
                  concat_node,
                  CreateConcatNodesForRetval(
                      sharding, arg_shapes[orig_arg_num].handle_type,
                      arg_shapes[orig_arg_num].handle_shape, replica,
                      orig_inputs, graph, device));
            }
            // Populate VariableWrite.
            VariableWrite& write = variable_writes->at(core_variable_writes[i]);
            write.value = concat_node;
            write.value_output = 0;
            write.predicate = compile_node;
            write.predicate_output = num_cores_per_replica + core + 1;

            continue;
          }

          // If this is a replicated variable, outputs on all cores will be the
          // same, and we only take the output from core 0 for the variable
          // update.
          if (IsReplicatedSharding(sharding) && core != 0) {
            continue;
          }
          VariableWrite& write = variable_writes->at(core_variable_writes[i]);
          write.value = node;
          write.value_output = num_outputs + i;
          write.predicate = compile_node;
          write.predicate_output = num_cores_per_replica + core + 1;
        }
      }
    }
  }

  for (Node* node : to_be_removed_nodes) {
    graph->RemoveNode(node);
  }
  return absl::OkStatus();
}  // NOLINT(readability/fn_size)

/* static */ absl::Status
DistributedTPURewritePass::CopyOutsideCompilationNodes(
    int replica_index, const std::vector<Node*>& outside_compilation_nodes,
    const DeviceNameUtils::ParsedName& tpu_device,
    const DeviceNameUtils::ParsedName& partial_device,
    NodeToNodeReplicasMap* node_images, Graph* graph) {
  for (Node* node : outside_compilation_nodes) {
    NodeDef image_def = node->def();
    MergeDebugInfo(NodeDebugInfo(node->def()), &image_def);
    const std::string suffix = absl::StrCat("/R", replica_index);
    // In addition to node name, make the frame name unique to avoid multiple
    // LoopCond nodes in one frame.
    TF_RETURN_IF_ERROR(
        AddPrefixAndSuffixToNode("" /* prefix */, suffix, &image_def));
    TF_ASSIGN_OR_RETURN(Node * image, graph->AddNode(image_def));
    image->AddAttr(kXlaReplicaIdAttrName, replica_index);
    if (HasNodeAttr(image->def(), kXlaHasHostTransferAttrName)) {
      TF_RETURN_IF_ERROR(
          SetNodeDeviceForTPUCommunication(tpu_device, DEVICE_CPU, image));
    } else {
      const std::string& original_device_string =
          node->assigned_device_name().empty() ? node->requested_device()
                                               : node->assigned_device_name();
      DeviceNameUtils::ParsedName device;
      TF_RET_CHECK(
          DeviceNameUtils::ParseFullName(original_device_string, &device));
      // If the requested device can be merged with the replica's host device,
      // then do so. For example, if the requested device is "/CPU:0" or
      // "/GPU:0" then it will be placed on the CPU/GPU of the host where this
      // replica is running. But if the requested device is
      // "/task:3/replica:2/CPU:0" then it will be placed on that task/replica.
      if (DeviceNameUtils::IsSpecification(device, partial_device)) {
        TF_RETURN_IF_ERROR(
            DeviceNameUtils::MergeDevNames(&device, partial_device));
      }
      image->set_requested_device(DeviceNameUtils::ParsedNameToString(device));
    }
    std::vector<Node*>& node_image_vector = (*node_images)[node];
    node_image_vector.resize(replica_index + 1);
    node_image_vector[replica_index] = image;
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::ReplicateOutsideCompilationNodes(
    const std::vector<std::vector<std::string>>& tf_device_assignment,
    const HostComputeCoreMap& host_compute_core,
    const OutsideCompilationNodeMap& outside_compilation_nodes,
    NodeToNodeReplicasMap* node_images, Graph* graph) {
  // Iterate over replicas.
  for (int i = 0; i < tf_device_assignment.size(); ++i) {
    const auto& core_devices = tf_device_assignment[i];
    for (const auto& oc_cluster_iter : outside_compilation_nodes) {
      const std::string& oc_cluster_name = oc_cluster_iter.first;
      const auto& oc_cluster_nodes = oc_cluster_iter.second;
      // We previously validated that host_compute_core contains an entry for
      // each cluster.
      int core = host_compute_core.at(oc_cluster_name);
      TF_RET_CHECK(core >= 0 && core < core_devices.size());
      // tpu_device is the device the HostCompute XLA Op for this cluster runs
      // on.
      DeviceNameUtils::ParsedName tpu_device;
      TF_RET_CHECK(
          DeviceNameUtils::ParseFullName(core_devices[core], &tpu_device));
      // partial_device contains the replica and task but not the type.
      DeviceNameUtils::ParsedName partial_device = tpu_device;
      partial_device.has_type = false;
      partial_device.has_id = false;

      if (tf_device_assignment.size() == 1) {
        // With a single replica don't copy any nodes just put the original
        // nodes into the image map. We leave the device placement alone, except
        // that we have to fill in the correct core for the host send and
        // receive nodes.
        for (Node* node : oc_cluster_nodes) {
          (*node_images)[node] = {node};
          node->AddAttr(kXlaReplicaIdAttrName, 0);
          if (HasNodeAttr(node->def(), kXlaHasHostTransferAttrName)) {
            TF_RETURN_IF_ERROR(
                SetNodeDeviceForTPUCommunication(tpu_device, DEVICE_CPU, node));
          }
        }
      } else {
        // Iterate over outside_compilation clusters in this computation, adding
        // all the nodes with appropriate device assignments.
        TF_RETURN_IF_ERROR(
            CopyOutsideCompilationNodes(i, oc_cluster_nodes, tpu_device,
                                        partial_device, node_images, graph));
      }
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::CopyOutsideCompilationEdges(
    const std::vector<Node*>& outside_compilation_nodes,
    const NodeToNodeReplicasMap& node_images,
    const std::unordered_map<std::string, Node*> outside_compilation_inputs,
    Graph* graph) {
  for (Node* node : outside_compilation_nodes) {
    const auto& images = node_images.at(node);
    // Make a copy of all edges and iterate on "in_edges", because we might
    // remove edges when iteratating through them.
    std::vector<const Edge*> in_edges(node->in_edges().begin(),
                                      node->in_edges().end());
    for (const Edge* edge : in_edges) {
      Node* src = edge->src();
      const auto iter = node_images.find(src);
      if (iter == node_images.end()) {
        if (images.size() > 1) {
          // The source node is a 'normal' node not part of any
          // rewrite. Broadcast the value to all replicas. (If images.size() ==
          // 1 the cluster is not replicated and we can leave the original edge
          // in place.)
          for (Node* dst : images) {
            graph->AddEdge(src, edge->src_output(), dst, edge->dst_input());
          }
        }
        continue;
      }

      // The source node is a replicated outside_compilation node.
      const auto& src_images = iter->second;
      if (src_images.size() != images.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Graph contains an edge from node ", src->name(),
            " in an outside_compilation block replicated ", src_images.size(),
            " ways to node ", node->name(),
            " in an outside_compilation block replicated ", images.size(),
            " ways. Replication factors must match. Leave a comment on "
            "tracking bug b/76419636 if you need this to be supported."));
      }
      bool is_lifted_arg;
      std::string outside_compilation_cluster;
      if (GetNodeAttr(src->def(), kXlaIsLiftedArgAttrName, &is_lifted_arg)
              .ok() &&
          GetNodeAttr(src->def(), kOutsideCompilationAttr,
                      &outside_compilation_cluster)
              .ok()) {
        const auto input_iter =
            outside_compilation_inputs.find(outside_compilation_cluster);
        TF_RET_CHECK(input_iter != outside_compilation_inputs.end());
        TF_RET_CHECK(input_iter->second->type_string() == "IdentityN");
        int dst_input = edge->dst_input();
        if (src_images.size() == 1) {
          graph->RemoveEdge(edge);
        }
        for (int i = 0; i < src_images.size(); ++i) {
          graph->AddEdge(input_iter->second, i, images[i], dst_input);
        }
        continue;
      }

      bool is_placeholder_for_arg;
      std::string outside_compilation_input_attr;
      if (GetNodeAttr(src->def(), kXlaIsPlaceholderForArg,
                      &is_placeholder_for_arg)
              .ok() &&
          GetNodeAttr(src->def(), kXlaOutsideCompilationInputsAttrName,
                      &outside_compilation_input_attr)
              .ok()) {
        const auto input_iter =
            outside_compilation_inputs.find(outside_compilation_input_attr);
        TF_RET_CHECK(input_iter != outside_compilation_inputs.end());
        TF_RET_CHECK(input_iter->second->type_string() == "IdentityN");
        int dst_input = edge->dst_input();
        if (src_images.size() == 1) {
          graph->RemoveEdge(edge);
        }
        for (int i = 0; i < src_images.size(); ++i) {
          graph->AddEdge(input_iter->second, i, images[i], dst_input);
        }
        continue;
      }

      if (images.size() > 1) {
        // If images.size() == 1 neither cluster is replicated and we can
        // leave the original edges in place.
        for (int i = 0; i < src_images.size(); ++i) {
          graph->AddEdge(src_images[i], edge->src_output(), images[i],
                         edge->dst_input());
        }
      }
    }
    for (const Edge* edge : node->out_edges()) {
      Node* dst = edge->dst();
      const auto iter = node_images.find(dst);
      if (iter == node_images.end()) {
        // The source node is a 'normal' node not part of any rewrite.
        if (edge->IsControlEdge()) {
          // Make the dst node have a control dependency on every replica.
          if (images.size() > 1) {
            for (int i = 0; i < images.size(); ++i) {
              graph->AddControlEdge(images[i], dst);
            }
          }
          // else the cluster is not replicated so we can leave the original
          // edge in place.
        } else {
          // The edge
          // is only valid if the outside_compilation block is not replicated.
          if (images.size() > 1) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Graph contains an edge from node ", node->name(),
                " in an outside_compilation block replicated ", images.size(),
                " ways to node ", dst->name(),
                " that is not part of an outside_compilation block. Edges from "
                "outside_compilation to regular graph nodes are only supported "
                "for replication factors of 1. Leave a comment on tracking bug "
                "b/76419636 if you need this to be supported."));
          }
          // else the cluster is not replicated so we can leave the original
          // edge in place.
        }
      }
      // The case where src and dst are both in node_images is covered elsewhere
      // when iterating over in_edges of dst.
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::ReplicateOutsideCompilationEdges(
    const OutsideCompilationNodeMap& outside_compilation_nodes,
    const NodeToNodeReplicasMap& node_images,
    const std::unordered_map<std::string, Node*> outside_compilation_inputs,
    Graph* graph) {
  for (const auto& oc_cluster_iter : outside_compilation_nodes) {
    TF_RETURN_IF_ERROR(
        CopyOutsideCompilationEdges(oc_cluster_iter.second, node_images,
                                    outside_compilation_inputs, graph));
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::RemoveOutsideCompilationNodes(
    const NodeToNodeReplicasMap& node_images, Graph* graph) {
  for (const auto& iter : node_images) {
    if (iter.second.size() > 1) {
      // The cluster was replicated so remove the original node.
      Node* node = iter.first;
      graph->RemoveNode(node);
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::LowerOutsideCompilationFunctionalNodes(
    Graph* g, FunctionLibraryDefinition& flib_def,
    const TPUReplicateDeviceNamesMapping& tpu_replicate_device_names_mapping) {
  bool modified = false;
  do {
    std::vector<Node*> nodes_to_lower;
    for (Node* n : g->op_nodes()) {
      if (!HasNodeAttr(n->def(), kOutsideCompilationAttr)) {
        continue;
      }

      if (n->IsWhileNode() || n->IsIfNode() || IsFunctionCall(flib_def, *n)) {
        // Only lower functional ops with DT_RESOURCE input, because otherwise
        // placer will complain. For normal cases, lowering will cause slowdown
        // when related functions are huge (b/139037679).
        bool has_resource_input = false;
        for (const Edge* e : n->in_edges()) {
          if (!e->IsControlEdge() &&
              e->src()->output_type(e->src_output()) == DT_RESOURCE) {
            has_resource_input = true;
            break;
          }
        }
        if (has_resource_input) {
          nodes_to_lower.push_back(n);
        }
      }
    }

    modified = !nodes_to_lower.empty();

    auto lower_functional_node = [&flib_def, &g](Node* n) -> absl::Status {
      // Clear device assignment. Otherwise all lowered nodes will have
      // device assignment, which is not what we want.
      n->set_requested_device("");

      int replica_id;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->def(), kXlaReplicaIdAttrName, &replica_id));

      std::string outside_compilation_attr;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), kOutsideCompilationAttr,
                                     &outside_compilation_attr));

      // There are two different kinds of functional outside compilation nodes:
      // 1. Nodes that are in outside compilation blocks already. They are
      //    generated by FunctionalizeControlFlowForXlaPass, and only have
      //    attribute kOutsideCompilationAttr.
      // 2. Mirrored control flow built for outside compilation in functional
      //    nodes. They are generated by ExtractOutsideCompilationPass, and have
      //    both kOutsideCompilationAttr and kXlaHasHostTransferAttrName.
      // When lowering them, they need to be treated differently.
      // For 1), their body functions are always V1 functions written by users,
      // and their "control outputs" are control inputs of _Retval nodes. They
      // should be lowered as V1 functions.
      // For 2), we always add necessary "control outputs"
      // (_XlaRecvAtHost/_XlaSendAtHost nodes) to "control_ret" field in their
      // FunctionDef's. They should be lowered as V2 functions.
      bool is_host_side_mirrored_control_flow =
          HasNodeAttr(n->def(), kXlaHasHostTransferAttrName);

      int num_node_ids = g->num_node_ids();
      bool is_call_node = IsFunctionCall(flib_def, *n);
      if (n->IsWhileNode()) {
        TF_RETURN_IF_ERROR(RewriteWhileNode(n, g, &flib_def,
                                            /*keep_node_fetchable=*/false));
      } else if (n->IsIfNode()) {
        TF_RETURN_IF_ERROR(RewriteIfNode(n, g, /*keep_node_fetchable=*/false));
      } else {
        TF_RET_CHECK(is_call_node);
        // See comments for "is_host_side_mirrored_control_flow" above.
        // If this is a node that's in outside compilation block, lower it as
        // V1 function. This is controlled by removing
        // kLowerAsMultiDeviceFunctionAttr from the node.
        if (!is_host_side_mirrored_control_flow) {
          n->ClearAttr(LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr);
        } else {
          n->ClearAttr(LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr);
          n->AddAttr(LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr,
                     true);
        }
        TF_RETURN_IF_ERROR(
            RewriteFunctionCallNode(n, g, flib_def,
                                    /*keep_caller_fetchable=*/false));
      }

      for (int i = num_node_ids; i < g->num_node_ids(); i++) {
        Node* node = g->FindNodeId(i);
        if (!node) {
          continue;
        }

        if (!is_call_node && is_host_side_mirrored_control_flow &&
            IsFunctionCall(flib_def, *node)) {
          // For If/While nodes, if they are host side mirrored control flow,
          // mark their body function calls with kXlaHasHostTransferAttrName
          // attribute to make sure we lower them as V2 function.
          node->AddAttr(kXlaHasHostTransferAttrName, true);
        }

        if (IsFunctionCall(flib_def, *node) || node->IsWhileNode() ||
            node->IsIfNode()) {
          // Set kOutsideCompilationAttr attribute so we lower these
          // nested function call nodes later.
          node->AddAttr(kOutsideCompilationAttr, outside_compilation_attr);
          // Set kXlaReplicaIdAttrName attribute so we know replica id when we
          // lower this function call node.
          node->AddAttr(kXlaReplicaIdAttrName, replica_id);
        } else if (node->type_string() == "_XlaRecvAtHost" ||
                   node->type_string() == "_XlaSendFromHost") {
          // For "_XlaRecvAtHost" and "_XlaSendFromHost" nodes, make sure they
          // have kXlaReplicaIdAttrName attribute so later we know which host
          // device to assign.
          node->AddAttr(kXlaReplicaIdAttrName, replica_id);
        }
      }
      return absl::OkStatus();
    };

    for (Node* n : nodes_to_lower) {
      TF_RETURN_IF_ERROR(lower_functional_node(n));
    }
  } while (modified);

  // Set device for all _XlaRecvAtHost and _XlaSendFromHost nodes.
  for (Node* n : g->op_nodes()) {
    if (n->type_string() != "_XlaRecvAtHost" &&
        n->type_string() != "_XlaSendFromHost") {
      continue;
    }

    std::string replicate;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), kTPUReplicateAttr, &replicate));
    auto iter = tpu_replicate_device_names_mapping.find(replicate);
    TF_RET_CHECK(iter != tpu_replicate_device_names_mapping.end());
    const auto& tpu_device_names = iter->second;

    int replica_id;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(n->def(), kXlaReplicaIdAttrName, &replica_id));
    TF_RET_CHECK(replica_id < tpu_device_names.size());
    const std::string& tpu_device_name = tpu_device_names[replica_id][0];
    std::string host_device_name;
    TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
        tpu_device_name, &host_device_name));
    n->set_assigned_device_name(host_device_name);
    // We may run TPU rewrite passes again on the subgraphs of the resulting
    // graph. Clear kTPUReplicateAttr and kOutsideCompilationAttr for
    // "_XlaRecvAtHost" nodes and "_XlaSendFromHost" nodes, in order to make
    // sure that TPU rewrite passes take no effect on host-side subgraphs for
    // outside compilation.
    n->ClearAttr(kTPUReplicateAttr);
    n->ClearAttr(kOutsideCompilationAttr);
  }

  // Remove IdentityN nodes generated for outside compilation. IdentityN is
  // exempt from resource edge colocation, but here we do need input and output
  // for these IdentityN nodes to be colocated.
  std::vector<Node*> identityn_nodes;
  for (Node* n : g->op_nodes()) {
    if (n->type_string() == "IdentityN" &&
        HasNodeAttr(n->def(), kXlaOutsideCompilationInputsAttrName)) {
      identityn_nodes.push_back(n);
    }
  }
  for (Node* n : identityn_nodes) {
    std::vector<const Edge*> out_edges(n->out_edges().begin(),
                                       n->out_edges().end());
    for (const Edge* e : out_edges) {
      if (e->IsControlEdge()) {
        continue;
      }

      int src_output = e->src_output();
      const Edge* input_edge;
      TF_RETURN_IF_ERROR(n->input_edge(src_output, &input_edge));
      Node* dst = e->dst();
      int dst_input = e->dst_input();
      g->RemoveEdge(e);
      g->AddEdge(input_edge->src(), input_edge->src_output(), dst, dst_input);
    }
    g->RemoveNode(n);
  }

  return absl::OkStatus();
}

/* static */ absl::Status DistributedTPURewritePass::ParseHostComputeCores(
    const Node& replicate_node,
    const OutsideCompilationNodeMap& outside_compilation_nodes,
    HostComputeCoreMap* host_compute_core) {
  std::vector<std::string> hc_core_string;
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(), "host_compute_core",
                                 &hc_core_string));
  TF_RETURN_IF_ERROR(
      ParseHostComputeCoreList(hc_core_string, host_compute_core));
  for (const auto& iter : outside_compilation_nodes) {
    const std::string& oc_cluster_name = iter.first;
    if (host_compute_core->find(oc_cluster_name) == host_compute_core->end()) {
      // By default put host compute Ops on replicated core 0.
      (*host_compute_core)[oc_cluster_name] = 0;
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status DistributedTPURewritePass::GetDeviceTopology(
    const DeviceSet& device_set, const Node& replicate_node, int* num_replicas,
    int* num_cores_per_replica, int* num_tasks,
    std::vector<std::vector<std::string>>* tf_device_assignment,
    std::vector<int>* devices_to_lock,
    std::unique_ptr<xla::DeviceAssignment>* xla_device_assignment,
    std::string* tpu_compilation_device) {
  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node.attrs(), "num_replicas", num_replicas));
  if (*num_replicas < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_replicas must be >= 1, got ", *num_replicas));
  }

  // Find the set of TPU devices in the TF job.
  // Indexed by [task number][tpu device number].
  std::vector<std::vector<Device*>> tpu_devices;
  int num_tpus_per_task;
  TF_RETURN_IF_ERROR(GetTPUDeviceNames(replicate_node.requested_device(),
                                       device_set, tpu_compilation_device,
                                       &num_tpus_per_task, &tpu_devices));
  *num_tasks = tpu_devices.size();

  std::string topology;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node.attrs(), "topology", &topology));
  TF_RETURN_IF_ERROR(GetNodeAttr(
      replicate_node.attrs(), "num_cores_per_replica", num_cores_per_replica));
  std::vector<int> device_assignment;
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(), "device_assignment",
                                 &device_assignment));

  // TODO(cwhipkey): since we can control multiple pods of different shapes
  // from a single worker, it may be desirable to propagate the remote device
  // information around (e.g., in DeviceAttributes). This can lead to the mesh
  // topology proto being leaked to cloud TPU users (e.g. through GetStatus
  // calls); this may be okay, but to be conservative, just assume that the
  // master session has the proper flags set.

  // The TPU system may be uninitialized yet, but we can still retrieve the
  // TPU topology even with an uninitialized TPU system via
  // TpuUtil_GetTopologyPtrFn.
  tpu::TpuTopologyExternal tpu_topology(
      stream_executor::tpu::OpsApiFn()->TpuUtil_GetTopologyPtrFn());
  TF_RET_CHECK(num_tpus_per_task ==
               tpu_topology.LogicalDevicesPerHost(kTensorCore));
  TF_RETURN_IF_ERROR(BuildDeviceAssignment(
      tpu_topology, num_tpus_per_task, tpu_devices, *num_replicas,
      *num_cores_per_replica, topology, device_assignment, tf_device_assignment,
      devices_to_lock, xla_device_assignment));

  return absl::OkStatus();
}

/* static */ absl::Status DistributedTPURewritePass::GetIOTypes(
    int num_replicas, const Node& replicate_node, FunctionLibraryRuntime* flr,
    Graph* graph, NameRangeMap* input_name_map, const NameAttrList** function,
    std::unique_ptr<Graph>* computation, DataTypeVector* arg_types,
    DataTypeVector* retval_types, ParameterInfo* params_info) {
  DataTypeVector input_types, broadcast_input_types, guaranteed_constant_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node.attrs(), "Tinputs", &input_types));
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(), "Tbroadcast_inputs",
                                 &broadcast_input_types));
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(),
                                 "Tguaranteed_constants",
                                 &guaranteed_constant_types));
  int num_distributed_vars;
  TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(),
                                 "num_distributed_variables",
                                 &num_distributed_vars));
  const int num_per_replica_inputs = input_types.size() - num_distributed_vars;

  if (num_per_replica_inputs % num_replicas != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of inputs to TPUReplicate (", num_per_replica_inputs,
        ") is not divisible by the number of replicas (", num_replicas, ")."));
  }

  int num_variables;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node.attrs(), "NumVariables", &num_variables));

  NameRangeMap output_name_map;
  TF_RETURN_IF_ERROR(NameRangesForNode(replicate_node, replicate_node.op_def(),
                                       input_name_map, &output_name_map));

  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node.attrs(), "computation", function));

  *computation = std::make_unique<Graph>(graph->op_registry());
  TF_RETURN_IF_ERROR(GetComputationForTPUReplicateOp(
      **function, flr, computation->get(), arg_types, retval_types));

  *params_info = ParameterInfo(
      num_replicas, num_per_replica_inputs / num_replicas, num_distributed_vars,
      broadcast_input_types.size(), num_variables,
      guaranteed_constant_types.size(), retval_types->size());

  if (arg_types->size() != params_info->NumInputsToEachReplica()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Computation argument to TPUReplicate has wrong number of "
                     "arguments. Expected ",
                     params_info->NumInputsToEachReplica(), " inputs, got ",
                     arg_types->size()));
  }
  if (replicate_node.num_outputs() != params_info->NumOutputsToHost()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Wrong number of outputs from TPUReplicate. Expected ",
                     params_info->NumOutputsToHost(), " outputs, got ",
                     replicate_node.num_outputs()));
  }
  if (enable_cross_replica_sharding_mirrored_variables_) {
    std::vector<int> mirrored_variable_indices;
    TF_RETURN_IF_ERROR(GetNodeAttr(replicate_node.attrs(),
                                   TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                                   &mirrored_variable_indices));
    for (int index : mirrored_variable_indices) {
      TF_RET_CHECK(params_info->IsPerReplicaArg(index) ||
                   params_info->IsDistributedArg(index))
          << "Mirrored variables not categorized as per-replica arguments, "
             "index: "
          << index;
      params_info->mutable_mirrored_variable_indices()->insert(index);
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status DistributedTPURewritePass::BuildSequencingNodes(
    const std::string& tpu_compilation_device, const Node& replicate_node,
    Graph* graph, Node** host_transfer_sequencer, Node** control_before,
    Node** control_after) {
  *host_transfer_sequencer = nullptr;

  TF_RETURN_IF_ERROR(
      BuildNoopNode(replicate_node,
                    graph->NewName(absl::StrCat(replicate_node.name(), "/",
                                                "control_before")),
                    /*device=*/"", graph, control_before));
  for (const Edge* e : replicate_node.in_edges()) {
    if (!e->IsControlEdge()) {
      continue;
    }
    Node* predecessor = e->src();
    if (predecessor->IsSource()) continue;
    if (predecessor->type_string() == "NoOp" &&
        predecessor->attrs().Find("_xla_host_transfer_sequencer") != nullptr) {
      // The node is the sequencer for host transfer operations. Its control
      // dependency needs to be placed after the execute node, not before.
      if (*host_transfer_sequencer != nullptr) {
        return absl::InternalError(absl::StrCat(
            "Replicate node ", replicate_node.name(),
            " has two transfer sequencer nodes: ",
            (*host_transfer_sequencer)->name(), " and ", predecessor->name()));
      }
      // Set the correct device to match the other sequencing nodes.
      predecessor->set_assigned_device_name(tpu_compilation_device);
      *host_transfer_sequencer = predecessor;
    } else {
      graph->AddControlEdge(predecessor, *control_before);
    }
  }

  TF_RETURN_IF_ERROR(BuildNoopNode(
      replicate_node,
      graph->NewName(absl::StrCat(replicate_node.name(), "/", "control_after")),
      /*device=*/tpu_compilation_device, graph, control_after));
  for (Node* successor : replicate_node.out_nodes()) {
    if (successor->attrs().Find("_xla_tail_outside_compilation") != nullptr) {
      graph->AddControlEdge(successor, *control_after);
    } else {
      graph->AddControlEdge(*control_after, successor);
    }
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::DealWithConstantsAndVariables(
    const Node& replicate_node, const NameRangeMap& input_name_map,
    Graph* graph, Node* host_transfer_sequencer, Node* control_before,
    Node* control_after, absl::Span<const VariableInput> variable_nodes,
    std::vector<Node*>* guaranteed_constant_nodes,
    std::vector<Node*>* variable_reads) {
  TF_RETURN_IF_ERROR(FindGuaranteedConstantInputs(
      replicate_node, input_name_map, guaranteed_constant_nodes));

  TF_RETURN_IF_ERROR(BuildVariableReads(variable_nodes, control_before, graph,
                                        variable_reads));
  // Add the control dependency from host transfer nodes.
  if (host_transfer_sequencer != nullptr) {
    graph->AddControlEdge(host_transfer_sequencer, control_after);
  }
  return absl::OkStatus();
}

/* static */ absl::Status
DistributedTPURewritePass::BuildCompilationStatusReturnNodes(
    Node* replicate_node, Node* compile_node,
    absl::Span<const int> devices_to_lock, Node** control_after_compilation,
    Node** multilock_acquire, Graph* graph) {
  const Edge* compilation_edge = nullptr;
  for (const auto* e : replicate_node->out_edges()) {
    if (e->IsControlEdge() &&
        e->dst()->type_string() == "TPUCompilationResult") {
      TF_RET_CHECK(compilation_edge == nullptr)
          << "Multiple compilation result nodes attached to the same replicate "
             "cluster.";
      compilation_edge = e;
    }
  }

  // TODO(jpienaar): This should be checked by default, current tests not using
  // this are ones that use the "abort upon successful compile flag" which will
  // be removed. Leaving this in until then.
  if (compilation_edge != nullptr) {
    Node* compilation_status = compilation_edge->dst();
    const AttrValue* compile_status_cluster_attr =
        compilation_status->attrs().Find(kTPUCompilationResultAttr);
    TF_RET_CHECK(compile_status_cluster_attr != nullptr);
    const std::string& compile_status_cluster =
        compile_status_cluster_attr->s();
    TF_RET_CHECK(!compile_status_cluster.empty());
    const AttrValue* replicate_cluster_attr =
        replicate_node->attrs().Find(kTPUReplicateAttr);
    TF_RET_CHECK(replicate_cluster_attr != nullptr);
    const std::string& replicate_cluster = replicate_cluster_attr->s();
    TF_RET_CHECK(!replicate_cluster.empty());
    TF_RET_CHECK(compile_status_cluster == replicate_cluster);

    TF_RETURN_IF_ERROR(
        ReplaceCompilationResultNodeWithIdentity(graph, &compilation_status));
    graph->AddEdge(compile_node, 0, compilation_status, 0);
  }

  NodeDef def;
  def.set_name(UniqueNodeName("tpu_compile_succeeded_assert", graph));
  // Create an op to assert that compilation succeeded. The alternative would
  // have been to have each execute op check and return an error.
  def.set_op("TPUCompileSucceededAssert");
  MergeDebugInfo(NodeDebugInfo(replicate_node->def()), &def);
  TF_ASSIGN_OR_RETURN(Node * compile_succeeded, graph->AddNode(def));
  compile_succeeded->set_assigned_device_name(
      compile_node->assigned_device_name());
  graph->AddEdge(compile_node, 0, compile_succeeded, 0);

  Node* last_node_before_sequencer = compile_succeeded;

  if (enable_multicore_locking_ && devices_to_lock.size() > 1) {
    // Add a lock node to acquire exclusive access to all the cores that will
    // execute this program. The lock is required to prevent deadlock or
    // incorrect results when running concurrent multi-core programs in the
    // same distributed runtime when there is no direct graph dependency
    // between the programs (either because they are run from different sessions
    // or because they are in the same graph, but have no control or data
    // dependencies to sequence them). Consider the case of two multi-core
    // computations A and B whose cores overlap and include cores X and Y. With
    // no locking and no graph dependencies it is possible that A's program
    // gets enqueued before B's on core X, while B's program gets enqueued
    // before A's on core Y. This will lead either to deadlock or to
    // incorrect results, since the runtime has no mechanism to re-sequence
    // the programs on the cores. By adding a multi-lock acquisition for all the
    // before any TPUExecute ops are run, and releasing it after they complete,
    // we ensure that the programs are enqueued on the cores in a consistent
    // order.
    //
    // There is a risk when computations are in the same graph, and include a
    // data dependency, that the lock acquisition could provoke deadlock.
    // Suppose that A must happen before B because B's input depends on A's
    // output. Then it is obviously necessary that A's lock acquisition must
    // happen before B's lock acquisition, and so we must ensure that there is
    // a graph dependency causing B's lock acquisition to be sequenced after A's
    // lock acquisition. Right now that dependency is satisfied because the
    // shape inference code cannot determine the shape of A's outputs, and so
    // B's compilation, which precedes B's lock acquisition, is always sequenced
    // after A's execution. If the shape inference is improved it will be
    // necessary to add an explicit control edge between dependent lock
    // acquisition ops.
    NodeDef lock_def;
    lock_def.set_name(graph->NewName(
        absl::StrCat(compile_node->name(), "/", "tpu_acquire_multilock")));
    lock_def.set_op("TpuMultilock");
    AddNodeAttr("lock_list", devices_to_lock, &lock_def);
    MergeDebugInfo(NodeDebugInfo(replicate_node->def()), &lock_def);
    TF_ASSIGN_OR_RETURN(*multilock_acquire, graph->AddNode(lock_def));
    (*multilock_acquire)
        ->set_assigned_device_name(compile_node->assigned_device_name());
    graph->AddControlEdge(compile_succeeded, *multilock_acquire);
    last_node_before_sequencer = *multilock_acquire;
  } else {
    *multilock_acquire = nullptr;
  }

  // Build a sequencing node for when compilation has completed.
  TF_RETURN_IF_ERROR(
      BuildNoopNode(*replicate_node,
                    graph->NewName(absl::StrCat(compile_node->name(), "/",
                                                "after_compilation")),
                    /*device=*/"", graph, control_after_compilation));
  graph->AddControlEdge(last_node_before_sequencer, *control_after_compilation);

  return absl::OkStatus();
}

// Updates the head and tail outside compiled nodes so that nodes have the
// correct device and removes the replication and outside compilation attributes
// so that these nodes do not trigger further graph optimization passes.
/* static */ absl::Status
DistributedTPURewritePass::UpdateHeadTailOutsideCompilation(
    const std::vector<std::vector<std::string>>& tf_device_assignment,
    const std::vector<Node*>& head_tail_outside_compilation_nodes) {
  for (Node* node : head_tail_outside_compilation_nodes) {
    int replica_id;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(node->def(), kXlaReplicaIdAttrName, &replica_id));
    // Since we set the device, this will now run on a task other than 0. We
    // clear the two following attributes so that we don't trigger encapsulation
    // again on the remote host (which will fail due to a missing
    // _TPUReplicateMetadata node for the cluster).
    for (const Edge* e : node->in_edges()) {
      // Resource consuming ops should colocate with its resource input.
      if (e->src()->IsArg() &&
          e->src()->output_type(e->src_output()) == DT_RESOURCE) {
        node->set_requested_device(tf_device_assignment[replica_id][0]);
      }
    }
    if (node->requested_device().empty()) {
      std::string cpu_device;
      TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
          tf_device_assignment[replica_id][0], &cpu_device));
      node->set_requested_device(cpu_device);
    }
    node->ClearAttr(kTPUReplicateAttr);
    node->ClearAttr(kOutsideCompilationAttr);
  }
  return absl::OkStatus();
}

// Performs the rewrite on a single TPUReplicate node.
/* static */ absl::Status DistributedTPURewritePass::RewriteTPUReplicateNode(
    const std::string& session_handle, const DeviceSet& device_set,
    Node* replicate_node, FunctionLibraryDefinition* flib_def,
    FunctionLibraryRuntime* flr, Node* host_compute_key_placeholder_node,
    const OutsideCompilationNodeMap& outside_compilation_nodes,
    const std::vector<Node*>& head_tail_outside_compilation_nodes,
    NodeToNodeReplicasMap* outside_compilation_node_images, Graph* graph,
    const GraphShapeInfo& shape_info,
    TPUReplicateDeviceNamesMapping* tpu_replicate_device_names_mapping,
    int64_t autotuner_thresh) {
  VLOG(2) << "Rewriting node " << replicate_node->name();

  // num_replicas and num_cores_per_replica are the 'virtual' replicas (copies
  // of the computation) and cores (virtual cores within computations) specified
  // by the user. They will be mapped to physical TPU cores below.
  int num_replicas;
  int num_cores_per_replica;
  int num_tasks;
  std::vector<std::vector<std::string>> tf_device_assignment;
  std::vector<int> devices_to_lock;
  std::unique_ptr<xla::DeviceAssignment> xla_device_assignment;
  std::string tpu_compilation_device;
  TF_RETURN_IF_ERROR(GetDeviceTopology(
      device_set, *replicate_node, &num_replicas, &num_cores_per_replica,
      &num_tasks, &tf_device_assignment, &devices_to_lock,
      &xla_device_assignment, &tpu_compilation_device));

  TF_RETURN_IF_ERROR(UpdateHeadTailOutsideCompilation(
      tf_device_assignment, head_tail_outside_compilation_nodes));

  std::string replicate;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(replicate_node->def(), kTPUReplicateAttr, &replicate));
  tpu_replicate_device_names_mapping->emplace(replicate, tf_device_assignment);

  NameRangeMap input_name_map;
  const NameAttrList* function;
  std::unique_ptr<Graph> computation;
  DataTypeVector arg_types, retval_types;
  ParameterInfo params_info;
  TF_RETURN_IF_ERROR(GetIOTypes(num_replicas, *replicate_node, flr, graph,
                                &input_name_map, &function, &computation,
                                &arg_types, &retval_types, &params_info));

  std::vector<InferredShape> arg_shapes, retval_shapes;
  TF_RETURN_IF_ERROR(GetArgAndRetvalShapes(
      shape_info, *replicate_node, params_info, &arg_shapes, &retval_shapes));

  TF_RETURN_IF_ERROR(ValidateCoreNumbers(*computation, num_cores_per_replica));

  std::vector<xla::OpSharding> arg_sharding;
  std::vector<bool> arg_fast_mem;
  std::vector<std::string> arg_names;
  std::vector<xla::OpSharding> retval_sharding;
  TF_RETURN_IF_ERROR(AssignArgsAndRetvalsToCores(
      num_cores_per_replica, params_info, arg_types, arg_shapes, retval_types,
      retval_shapes, *computation, replicate_node, flr,
      allow_xla_spmd_partition_, &arg_sharding, &arg_fast_mem, &retval_sharding,
      &arg_names));

  VLOG(1) << DumpGraphToFile("distributed_tpu_graph_to_replicate", *computation,
                             flib_def);

  GraphDef graph_def;
  graph->ToGraphDef(&graph_def);
  FunctionLibraryDefinition reachable_functions =
      flib_def->ReachableDefinitions(graph_def);
  uint64_t library_fingerprint;

  TF_RETURN_IF_ERROR(
      FingerprintFunctionLibrary(reachable_functions, library_fingerprint));
  VLOG(1) << "Fingerprint functions: "
          << absl::StrJoin(reachable_functions.ListFunctionNames(), ", ");
  VLOG(1) << "library_fingerprint: " << library_fingerprint;

  // Builds trigger nodes that put barriers around the expansion of
  // TPUReplicate. In particular, we must guarantee:
  // a) variable reads happen after all predecessors of the original
  //    TPUReplicate.
  // b) variable writes happen before all successors of the original
  //    TPUReplicate.
  // c) all replicas execute, even if output tensors are only requested from
  //    a subset of replicas. This is necessary both to ensure that variable
  //    updates happen, but also Send/Recv will deadlock if only one half of
  //    the communicating pair runs.
  Node* host_transfer_sequencer;
  Node* control_before;
  Node* control_after;
  TF_RETURN_IF_ERROR(BuildSequencingNodes(
      tpu_compilation_device, *replicate_node, graph, &host_transfer_sequencer,
      &control_before, &control_after));

  // Build a vector of variable nodes that are inputs.
  std::vector<VariableInput> variable_inputs;
  TF_RETURN_IF_ERROR(
      FindVariableInputs(*replicate_node, input_name_map, &variable_inputs));

  std::vector<Node*> guaranteed_constant_nodes;
  std::vector<Node*> variable_reads;
  TF_RETURN_IF_ERROR(DealWithConstantsAndVariables(
      *replicate_node, input_name_map, graph, host_transfer_sequencer,
      control_before, control_after, variable_inputs,
      &guaranteed_constant_nodes, &variable_reads));

  // Builds Shape nodes that compute the dynamic shapes of arguments whose
  // shapes are not statically known.
  std::vector<Node*> dynamic_shape_nodes;
  TF_RETURN_IF_ERROR(BuildDynamicShapeNodes(*replicate_node, arg_shapes,
                                            params_info, variable_reads, graph,
                                            &dynamic_shape_nodes));

  // Builds a TPUCompile node that compiles `clusters` on `compile_device`.
  Node* compile_node;
  TF_RETURN_IF_ERROR(BuildCompileNode(
      replicate_node, *function, library_fingerprint, params_info, arg_shapes,
      arg_types, guaranteed_constant_nodes, session_handle, arg_sharding,
      arg_fast_mem, arg_names, retval_sharding, num_cores_per_replica,
      /*compile_device=*/tpu_compilation_device, xla_device_assignment.get(),
      dynamic_shape_nodes, graph, &compile_node, autotuner_thresh));

  // Compilation must be sequenced after the control node if the TPU computation
  // in a control-flow construct, such as a loop.
  graph->AddControlEdge(control_before, compile_node);

  Node* control_after_compilation;
  Node* multilock_acquire;
  TF_RETURN_IF_ERROR(BuildCompilationStatusReturnNodes(
      replicate_node, compile_node, devices_to_lock, &control_after_compilation,
      &multilock_acquire, graph));

  std::vector<VariableWrite> variable_writes;
  TF_RETURN_IF_ERROR(BuildExecuteNodes(
      params_info, num_tasks, num_cores_per_replica, *replicate_node, arg_names,
      arg_types, arg_shapes, retval_types, arg_sharding, retval_sharding,
      tf_device_assignment, compile_node, variable_reads,
      control_after_compilation, control_after, multilock_acquire,
      &variable_writes, graph));
  bool contains_resource_write_op =
      ContainsResourceWriteOp(*graph, reachable_functions);

  VLOG(2) << "contains_resource_write_op: " << contains_resource_write_op;
  // Skip conditional write if there is no resource writing op inside TPU
  // computation.
  if (contains_resource_write_op) {
    TF_RETURN_IF_ERROR(BuildVariableWrites(variable_inputs, control_after,
                                           variable_writes, graph));
  }

  if (host_compute_key_placeholder_node != nullptr) {
    TF_RETURN_IF_ERROR(ConnectHostComputeNodes(
        compile_node, host_compute_key_placeholder_node, graph));
  }

  HostComputeCoreMap host_compute_core;
  TF_RETURN_IF_ERROR(ParseHostComputeCores(
      *replicate_node, outside_compilation_nodes, &host_compute_core));
  TF_RETURN_IF_ERROR(ReplicateOutsideCompilationNodes(
      tf_device_assignment, host_compute_core, outside_compilation_nodes,
      outside_compilation_node_images, graph));

  graph->RemoveNode(replicate_node);
  return absl::OkStatus();
}

// Adds sharded weight update optimization for each host training loop.
//
// For any host training loop found in the graph, TPUVariableReshard ops
// are inserted to match the best layout chosen by the XLA.
/* static */ absl::Status
DistributedTPURewritePass::PerformHostTrainingLoopOptimization(
    Graph* graph, FunctionLibraryDefinition* flib_def,
    FunctionLibraryRuntime* flr) {
  std::vector<tpu::HostTrainingLoopInfo> host_training_loops_info;
  absl::Status s = tpu::DetectHostTrainingLoop(
      /*current_function_name=*/nullptr,
      /*current_function_attr=*/nullptr, flib_def, graph, flr,
      &host_training_loops_info);
  if (!s.ok()) {
    VLOG(2) << "No valid host training loop found. Skipping sharded weight "
            << "update optimization.";
    return absl::OkStatus();
  }

  for (const auto& host_loop : host_training_loops_info) {
    const auto& function_name = host_loop.encapsulating_function_name;
    // `function_name` has value when host training loop is inside a
    // function call node. When host training loop is found inside a function
    // call node, then, in addition to adding TPUVariableReshard ops, function
    // library definition needs to be updated as well.
    if (function_name.has_value()) {
      const auto& function_attr = host_loop.encapsulating_function_attrs;
      TF_RET_CHECK(function_attr.has_value())
          << "Unable to find function attribute for function: "
          << *function_name;

      const FunctionDef* function_def = flib_def->Find(*function_name);
      TF_RET_CHECK(function_def)
          << "Unable to find function : " << *function_name;

      std::unique_ptr<FunctionBody> fbody;
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          *function_def, AttrSlice(&function_attr.value()), flib_def, &fbody));
      Graph* function_graph = fbody->graph;
      TF_RETURN_IF_ERROR(tpu::AddReshardOp(function_graph, host_loop));
      TF_RETURN_IF_ERROR(UpdateFunctionLibDefinition(*function_graph,
                                                     *function_name, flib_def));
    } else {
      TF_RETURN_IF_ERROR(tpu::AddReshardOp(graph, host_loop));
    }
  }
  return absl::OkStatus();
}

absl::Status
DistributedTPURewritePass::PlaceUnassignedDeviceNodesOnTPUIfPossible(
    Graph* graph) {
  PropagateDevices(CanAcceptTPUDevicePropagation, IsTpuDevice, graph);
  return absl::OkStatus();
}

absl::Status DistributedTPURewritePass::Run(
    const GraphOptimizationPassOptions& options) {
  absl::Status status = InternalRun(options);
  tsl::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::TF_XLA_BRIDGE, status);
  return status;
}

absl::Status DistributedTPURewritePass::InternalRun(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "DistributedTPURewritePass::Run";

  Graph* graph = options.graph->get();

  VLOG(1) << DumpGraphToFile("distributed_tpu_compilation_before", *graph,
                             options.flib_def);

  const auto* config = &options.session_options->config;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          nullptr, options.session_options->env, config,
          graph->versions().producer(), options.flib_def,
          config ? config->graph_options().optimizer_options()
                 : OptimizerOptions()));

  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  // This pass can only run in the session master, which should fill
  // in the device_set field to the options.
  TF_RET_CHECK(options.device_set != nullptr);

  // Find all the replicate nodes before mutating the graph.
  std::vector<Node*> replicate_nodes;
  // Map from compiled subgraph cluster name to the outside_compilation nodes in
  // that cluster.
  std::map<std::string, OutsideCompilationNodeMap> outside_compilation_nodes;
  std::map<std::string, std::vector<Node*>> head_tail_outside_compilation_nodes;
  TF_RETURN_IF_ERROR(FindTaggedNodes(graph, &replicate_nodes,
                                     &outside_compilation_nodes,
                                     &head_tail_outside_compilation_nodes));

  if (replicate_nodes.empty()) {
    // Remove unused TPUPartitionedInput nodes.
    for (Node* n : graph->nodes()) {
      if (_IsTPUPartitionedInput(n)) graph->RemoveNode(n);
    }
    VLOG(1) << DumpGraphToFile("distributed_tpu_compilation_after", *graph,
                               options.flib_def);
    VLOG(1) << "Replicate nodes are empty. DistributedTPURewritePass::Run() "
               "finished";
    return absl::OkStatus();
  }

  std::unordered_map<std::string, Node*> host_compute_key_placeholder_map;
  TF_RETURN_IF_ERROR(FindHostComputeKeyPlaceholderNodes(
      graph, replicate_nodes, &host_compute_key_placeholder_map));

  // This shape inference pass does not compute the shapes of outputs of
  // TPU computations. The concurrent multi-core locking implementation
  // *relies* on this behavior because it ensures that, if TPU computation B's
  // inputs depend on TPU computation A's outputs, then computation B's
  // compilation will be sequenced after A's execution, and this ensures that
  // locks are acquired in the correct order. If the shape inference is improved
  // to compute shapes of TPU computation outputs, it will be necessary to add
  // an explicit control edge between lock acquisitions for dependent
  // computations in order to avoid deadlock.
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(InferShapes(graph, /*arg_shapes=*/{},
                                 flr->GetFunctionLibraryDefinition(),
                                 &shape_info));
  int64_t autotuner_thresh = options.session_options->config.experimental()
                                 .xla_fusion_autotuner_thresh();

  NodeToNodeReplicasMap outside_compilation_node_images;
  TPUReplicateDeviceNamesMapping tpu_replicate_device_names_mapping;
  for (Node* node : replicate_nodes) {
    TF_RETURN_IF_ERROR(RewriteTPUReplicateNode(
        options.session_handle, *options.device_set, node, options.flib_def,
        flr, host_compute_key_placeholder_map[node->name()],
        outside_compilation_nodes[node->name()],
        head_tail_outside_compilation_nodes[node->name()],
        &outside_compilation_node_images, graph, shape_info,
        &tpu_replicate_device_names_mapping, autotuner_thresh));
  }

  // Place the padding nodes generated by dynamic padder on the correct devices.
  // TODO(rxsang): Place padding ops on TPUs in
  // PlaceUnassignedDeviceNodesOnTPUIfPossible function.
  TF_RETURN_IF_ERROR(SetPaddingNodesDevices(graph));

  std::unordered_map<std::string, Node*> outside_compilation_inputs;
  for (Node* n : graph->op_nodes()) {
    std::string lifted_arg_inputs_attr;
    if (n->type_string() == "IdentityN" &&
        GetNodeAttr(n->def(), kXlaOutsideCompilationInputsAttrName,
                    &lifted_arg_inputs_attr)
            .ok()) {
      outside_compilation_inputs[lifted_arg_inputs_attr] = n;
    }
  }
  for (const auto& iter : outside_compilation_nodes) {
    TF_RETURN_IF_ERROR(ReplicateOutsideCompilationEdges(
        iter.second, outside_compilation_node_images,
        outside_compilation_inputs, graph));
  }
  TF_RETURN_IF_ERROR(
      RemoveOutsideCompilationNodes(outside_compilation_node_images, graph));
  TF_RETURN_IF_ERROR(LowerOutsideCompilationFunctionalNodes(
      graph, *options.flib_def, tpu_replicate_device_names_mapping));

  TF_RETURN_IF_ERROR(PlaceUnassignedDeviceNodesOnTPUIfPossible(graph));
  VLOG(1) << DumpGraphToFile("distributed_tpu_compilation_after", *graph,
                             options.flib_def);
  VLOG(1) << "DistributedTPURewritePass::Run() finished";

  if (enable_cross_replica_sharding_mirrored_variables_) {
    VLOG(1) << "Starting host training loop optimization.";
    VLOG(1) << DumpGraphToFile("host_loop_optimization_before", *graph,
                               options.flib_def);
    TF_RETURN_IF_ERROR(
        PerformHostTrainingLoopOptimization(graph, options.flib_def, flr));
    VLOG(1) << DumpGraphToFile("host_loop_optimization_after", *graph,
                               options.flib_def);
    VLOG(1) << "Host training loop optimization finished.";
  }

  return absl::OkStatus();
}

bool DistributedTPURewritePass::distribute_vars_ = false;
bool DistributedTPURewritePass::allow_xla_spmd_partition_ = true;
bool DistributedTPURewritePass::
    replicate_inputs_outputs_by_default_for_xla_spmd_ = false;
bool DistributedTPURewritePass::
    enable_cross_replica_sharding_mirrored_variables_ = true;
bool DistributedTPURewritePass::enable_automatic_model_parallelism_ = false;
bool DistributedTPURewritePass::enable_xla_param_broadcast_ = true;
bool DistributedTPURewritePass::enable_multicore_locking_ = false;
bool DistributedTPURewritePass::use_nd_sharding_ops_ = false;

/*static*/ void DistributedTPURewritePass::SetDistributedTpuRewritePassOptions(
    bool distribute_vars, bool allow_xla_spmd_partition,
    bool replicate_inputs_outputs_by_default_for_xla_spmd,
    bool enable_cross_replica_sharding_mirrored_variables,
    bool enable_automatic_model_parallelism, bool enable_xla_param_broadcast,
    bool enable_multicore_locking, bool use_nd_sharding_ops) {
  distribute_vars_ = distribute_vars;
  allow_xla_spmd_partition_ = allow_xla_spmd_partition;
  replicate_inputs_outputs_by_default_for_xla_spmd_ =
      replicate_inputs_outputs_by_default_for_xla_spmd;
  enable_cross_replica_sharding_mirrored_variables_ =
      enable_cross_replica_sharding_mirrored_variables;
  enable_automatic_model_parallelism_ = enable_automatic_model_parallelism;
  enable_xla_param_broadcast_ = enable_xla_param_broadcast;
  enable_multicore_locking_ = enable_multicore_locking;
  use_nd_sharding_ops_ = use_nd_sharding_ops;
}

}  // namespace tensorflow
