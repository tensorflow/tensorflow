/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"

#include <fstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {
namespace {

const std::pair<int, int> kMinGPUArch = {7, 0};

const char kSuffix[] = "AutoMixedPrecision";
const char kCastToFp16[] = "CastToFp16";
const char kCastToFp32[] = "CastToFp32";

// Instances of this class represent unique type attribute identifiers within a
// node. It handles regular type attributes, list type attributes (where
// type_index is set to the index in the type list), and fixed types.
struct TypeAttrId {
  static const int kSingleType = -1;

  explicit TypeAttrId(const string& _attr_name, int _type_index = kSingleType)
      : attr_name(_attr_name),
        type_index(_type_index),
        fixed_type(DT_INVALID) {}

  explicit TypeAttrId(DataType _fixed_type)
      : attr_name(), type_index(kSingleType), fixed_type(_fixed_type) {}

  bool operator==(const TypeAttrId& other) const {
    return attr_name == other.attr_name && type_index == other.type_index &&
           fixed_type == other.fixed_type;
  }

  bool operator<(const TypeAttrId& other) const {
    return std::make_tuple(attr_name, type_index, fixed_type) <
           std::make_tuple(other.attr_name, other.type_index, other.fixed_type);
  }

  template <typename H>
  friend H AbslHashValue(H h, const TypeAttrId& ta) {
    return H::combine(std::move(h), ta.attr_name, ta.type_index, ta.fixed_type);
  }

  string DebugString() const {
    if (!attr_name.empty()) {
      if (type_index == kSingleType) {
        return attr_name;
      } else {
        return strings::StrCat(attr_name, "[", type_index, "]");
      }
    } else {
      return tensorflow::DataTypeString(fixed_type);
    }
  }

  string attr_name;
  // If attr_name is a list(type), this is the index into the list. Otherwise
  // this is kSingleType.
  int type_index;
  DataType fixed_type;
};

// Returns the data type of the given type attribute, or DT_INVALID if the type
// attribute is invalid.
DataType GetDataType(const NodeDef& node, const TypeAttrId& type_attr) {
  if (type_attr.attr_name.empty()) {
    return type_attr.fixed_type;
  }
  if (!node.attr().count(type_attr.attr_name)) {
    return DT_INVALID;
  }
  const AttrValue& attr_value = node.attr().at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    return attr_value.type();
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return DT_INVALID;
    }
    return attr_value.list().type(type_attr.type_index);
  }
}

// Sets the data type of the given type attribute. Returns false if the type
// attribute is invalid, otherwise true.
bool SetDataType(NodeDef* node, const TypeAttrId& type_attr, DataType type) {
  if (type_attr.attr_name.empty() || !node->attr().count(type_attr.attr_name)) {
    return false;
  }
  AttrValue& attr_value = node->mutable_attr()->at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    attr_value.set_type(type);
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return false;
    }
    attr_value.mutable_list()->set_type(type_attr.type_index, type);
  }
  return true;
}

std::vector<std::pair<int, int>> ArgDefIndexes(const NodeDef& node, int arg_idx,
                                               const OpDef::ArgDef& arg_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  if (!arg_def.type_list_attr().empty()) {
    int num_types = node.attr().at(arg_def.type_list_attr()).list().type_size();
    for (int type_idx = 0; type_idx < num_types; ++type_idx) {
      argdef_inds.push_back({arg_idx, type_idx});
    }
  } else {
    int num_repeat = 1;
    if (node.attr().count(arg_def.number_attr())) {
      num_repeat = node.attr().at(arg_def.number_attr()).i();
    }
    argdef_inds.insert(argdef_inds.end(), num_repeat, {arg_idx, -1});
  }
  return argdef_inds;
}

// Returns a pair (arg_index, type_index) for each input to the node, where
// arg_index is the index of the input_arg in op_def and type_index is the index
// of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> InputPortArgDefIndexes(const NodeDef& node,
                                                        const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.input_arg_size());  // Final size may differ.
  for (int arg_idx = 0; arg_idx < op_def.input_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.input_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}

// Returns a pair (arg_index, type_index) for each output to the node, where
// arg_index is the index of the output_arg in op_def and type_index is the
// index of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> OutputPortArgDefIndexes(const NodeDef& node,
                                                         const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.output_arg_size());  // Final size may differ.
  for (int arg_idx = 0; arg_idx < op_def.output_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.output_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}

TypeAttrId GetTypeAttrId(const OpDef::ArgDef& arg_def, int arg_type_index) {
  if (!arg_def.type_list_attr().empty()) {
    return TypeAttrId(arg_def.type_list_attr(), arg_type_index);
  } else if (!arg_def.type_attr().empty()) {
    return TypeAttrId(arg_def.type_attr());
  } else {
    return TypeAttrId(arg_def.type());
  }
}

std::vector<int> NonControlInputs(const NodeDef& node) {
  std::vector<int> pos;
  for (int i = 0; i < node.input_size(); i++) {
    if (!IsControlInput(node.input(i))) {
      pos.push_back(i);
    }
  }
  return pos;
}

// A utility class to lookup node type attributes and type attribute <->
// input/output port mappings.
class NodeTypeAttrMap {
 public:
  NodeTypeAttrMap() {}

  explicit NodeTypeAttrMap(const GraphDef& graph) { TF_CHECK_OK(Init(graph)); }

  Status Init(const GraphDef& graph) {
    if (graph_ != nullptr) {
      return errors::InvalidArgument("NodeTypeAttrMap is already initialized.");
    }
    graph_ = &graph;
    function_library_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
    for (const NodeDef& node : graph.node()) {
      TF_RETURN_IF_ERROR(AddNode(node));
    }
    return Status::OK();
  }

  bool is_initialized() const { return graph_ != nullptr; }

  // Returns the set of all type attributes in the given node.
  absl::flat_hash_set<TypeAttrId> GetTypeAttrs(const NodeDef& node) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    absl::flat_hash_set<TypeAttrId> type_attrs;
    const auto iter = type2io_.find(&node);
    CHECK(iter != type2io_.end());  // Crash Ok
    for (const auto& key_value : iter->second) {
      type_attrs.insert(key_value.first);
    }
    return type_attrs;
  }

  const absl::flat_hash_set<int>& GetInputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    return type2io_.at(&node).at(type_attr).first;
  }

  const absl::flat_hash_set<int>& GetOutputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    return type2io_.at(&node).at(type_attr).second;
  }

  TypeAttrId GetInputTypeAttr(const NodeDef& node, int port) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    auto type_vec = io2type_.at(&node).first;
    CHECK_GE(port, 0);                // Crash Ok
    CHECK_LT(port, type_vec.size());  // Crash Ok
    return type_vec[port];
  }

  TypeAttrId GetOutputTypeAttr(const NodeDef& node, int port) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    auto type_vec = io2type_.at(&node).second;
    CHECK_GE(port, 0);                // Crash Ok
    CHECK_LT(port, type_vec.size());  // Crash Ok
    return type_vec[port];
  }

 private:
  Status AddNode(const NodeDef& node) {
    const OpDef* op_def_ptr = nullptr;
    TF_RETURN_IF_ERROR(function_library_->LookUpOpDef(node.op(), &op_def_ptr));
    const OpDef& op_def = *op_def_ptr;
    auto& type2io_entry = type2io_[&node];
    auto& io2type_entry = io2type_[&node];
    auto input_arg_inds = InputPortArgDefIndexes(node, op_def);
    if (NonControlInputs(node).size() != input_arg_inds.size()) {
      return errors::InvalidArgument(
          "Expected ", node.op(), " node ", node.name(), " to have ",
          input_arg_inds.size(), " non-control input(s), but got ",
          node.input_size());
    }
    // Note that the mappings generated here include inputs/outputs with fixed
    // types. This makes the mappings complete (all inputs and outputs are
    // included), and allows the graph rewriter to propagate black paint
    // from/through ops with fixed types.
    io2type_entry.first.reserve(input_arg_inds.size());
    for (int i = 0; i < static_cast<int>(input_arg_inds.size()); ++i) {
      const auto& arg_inds = input_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.input_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].first.insert(i);
      io2type_entry.first.push_back(type_attr);
    }

    auto output_arg_inds = OutputPortArgDefIndexes(node, op_def);
    io2type_entry.second.reserve(output_arg_inds.size());
    for (int i = 0; i < static_cast<int>(output_arg_inds.size()); ++i) {
      const auto& arg_inds = output_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.output_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].second.insert(i);
      io2type_entry.second.push_back(type_attr);
    }

    // Also ensure that type attributes that aren't associated with any inputs
    // or outputs (e.g., StackV2's elem_type) are added to the map.
    for (const auto& attr : node.attr()) {
      const string& attr_name = attr.first;
      if (!attr_name.empty() && attr_name[0] == '_') continue;
      const AttrValue& attr_value = attr.second;
      const OpDef::AttrDef* attr_def = FindAttr(attr_name, op_def);
      if (!attr_def) {
        return errors::InvalidArgument("AttrDef not found for attribute ",
                                       attr_name, " of node ", node.name());
      }
      if (attr_def->type() == "type") {
        type2io_entry[TypeAttrId(attr_name)];
      } else if (attr_def->type() == "list(type)") {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          type2io_entry[TypeAttrId(attr_name, i)];
        }
      }
    }
    return Status::OK();
  }

  // WARN: `graph_` must outlive this object (node pointers must remain valid).
  const GraphDef* graph_ = nullptr;  // do not own
  std::unique_ptr<FunctionLibraryDefinition> function_library_;

  typedef absl::flat_hash_set<int> IntSet;
  // Maps a type attr id -> (input port set, output port set)
  typedef absl::flat_hash_map<TypeAttrId, std::pair<IntSet, IntSet>> Type2IOMap;
  // Maps a node -> type attr mapping
  absl::flat_hash_map<const NodeDef*, Type2IOMap> type2io_;
  // Maps a port -> type attr id
  typedef std::vector<TypeAttrId> TypeAttrIdVec;
  // Maps a node -> (input port mapping, output port mapping)
  absl::flat_hash_map<const NodeDef*, std::pair<TypeAttrIdVec, TypeAttrIdVec>>
      io2type_;
};

struct NodeTypeId {
  NodeTypeId(const NodeDef* _node, const TypeAttrId& _type_attr)
      : node(_node), type_attr(_type_attr) {}

  const NodeDef* node;
  TypeAttrId type_attr;

  bool operator==(const NodeTypeId& other) const {
    return node == other.node && type_attr == other.type_attr;
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodeTypeId& nt) {
    return H::combine(std::move(h), nt.node, nt.type_attr);
  }
};

struct NodeTypeIdEdge {
  NodeTypeIdEdge(const NodeTypeId& _src, const NodeTypeId& _dst)
      : src(_src), dst(_dst) {}
  NodeTypeId src;
  NodeTypeId dst;
};

// TODO(benbarsdell): Investigate whether the existing GraphTopologyView can be
// used instead of this modified version.
// This is just like GraphTopologyView but with (NodeDef, TypeAttrId) pairs as
// the vertices instead of just NodeDef.
// For example, if node A has output A:0 with TypeAttrId 'T', and node B has
// input B:0 with TypeAttrId 'U', and input B:0 connects to output A:0, there
// will be an edge from (A, T) to (B, U).
class GraphTypeTopologyView {
 public:
  GraphTypeTopologyView() = default;
  explicit GraphTypeTopologyView(bool skip_invalid_edges)
      : skip_invalid_edges_(skip_invalid_edges) {}

  // Initialize graph topology view from the graph. It's possible to pass
  // additional edges that do not exist in a graph, but must be respected when
  // computing graph topology. Example: Tensorflow runtime allows concurrent
  // execution of dequeue/enqueue ops from the same queue resource, but we might
  // want to enforce ordering between them for the purpose of graph analysis.
  Status InitializeFromGraph(const GraphDef& graph,
                             const NodeTypeAttrMap& node_type_map,
                             absl::Span<const NodeTypeIdEdge> ephemeral_edges);
  Status InitializeFromGraph(const GraphDef& graph,
                             const NodeTypeAttrMap& node_type_map);

  bool is_initialized() const { return graph_ != nullptr; }
  int num_nodes() const { return num_nodes_; }
  const GraphDef* graph() const { return graph_; }

  // Returns true iff the node exists in the underlying graph.
  bool HasNode(absl::string_view node_name, const TypeAttrId& type_attr) const;

  // Finds a node by name or returns `nullptr` if it's not in the graph.
  const NodeTypeId* GetNode(absl::string_view node_name,
                            const TypeAttrId& type_attr) const;
  // Returns a node corresponding to the given node index.
  const NodeTypeId* GetNode(int node_idx) const;

  // Returns a node index for the given node name, if the name exists in the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(absl::string_view node_name,
                                         const TypeAttrId& type_attr) const;
  // Returns a node index for the given node, if the node belongs to the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(const NodeTypeId& node) const;

  // Returns all the node indexes that are in the direct fanin of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 4>& GetFanin(int node_idx) const;
  // Returns all the node indexes that are in the direct fanout of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 2>& GetFanout(int node_idx) const;

 private:
  // The key type used to uniquely identify a type attribute on a node.
  struct NodeTypeKey : public std::pair<absl::string_view, TypeAttrId> {
    typedef std::pair<absl::string_view, TypeAttrId> Base;

    // Inherit the set of constructors.
    using Base::pair;

    template <typename H>
    friend H AbslHashValue(H h, const NodeTypeKey& nt) {
      return H::combine(std::move(h), nt.first, nt.second);
    }
  };

  // If true, all invalid edges and inputs (srd, dst or input node not found in
  // a graph) will be skipped, otherwise initialization will fail with error.
  bool skip_invalid_edges_ = false;

  // WARN: `graph_` must outlive this object and graph nodes must not be
  // destructed, because node names captured with absl::string_view.
  const GraphDef* graph_ = nullptr;  // do not own
  int num_nodes_ = 0;
  std::vector<NodeTypeId> node_type_attrs_;
  absl::flat_hash_map<absl::string_view, int> node_name_to_index_;
  absl::flat_hash_map<NodeTypeKey, int> node_type_name_to_index_;

  std::vector<absl::InlinedVector<int, 4>> fanins_;
  std::vector<absl::InlinedVector<int, 2>> fanouts_;

  // We need a valid reference to return from GetFanin/GetFanout if the
  // `node_idx` argument is outside of the [0, num_nodes_) range.
  absl::InlinedVector<int, 4> empty_fanin_;
  absl::InlinedVector<int, 2> empty_fanout_;
};

template <typename T>
inline void SortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

Status GraphTypeTopologyView::InitializeFromGraph(
    const GraphDef& graph, const NodeTypeAttrMap& node_type_map,
    absl::Span<const NodeTypeIdEdge> ephemeral_edges) {
  if (graph_ != nullptr) {
    return errors::InvalidArgument(
        "GraphTypeTopologyView is already initialized.");
  }

  graph_ = &graph;
  int num_nodedefs = graph.node_size();
  node_name_to_index_.rehash(num_nodedefs);

  // Build maps from name to index.
  node_type_attrs_.reserve(num_nodedefs);         // Only approximate.
  node_type_name_to_index_.rehash(num_nodedefs);  // Only approximate.
  for (int node_idx = 0; node_idx < num_nodedefs; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    node_name_to_index_.emplace(node.name(), node_idx);

    for (const TypeAttrId& type_attr : node_type_map.GetTypeAttrs(node)) {
      int node_type_idx = node_type_attrs_.size();
      node_type_name_to_index_.emplace(NodeTypeKey(node.name(), type_attr),
                                       node_type_idx);
      node_type_attrs_.emplace_back(&node, type_attr);
    }
  }
  num_nodes_ = node_type_attrs_.size();
  fanins_.resize(num_nodes_);
  fanouts_.resize(num_nodes_);

  // 1. Add ephemeral edges to the adjacency lists.
  for (const NodeTypeIdEdge& edge : ephemeral_edges) {
    const auto src = node_name_to_index_.find(edge.src.node->name());
    const bool valid_src = src != node_name_to_index_.end();

    if (!valid_src) {
      const string error_message =
          absl::StrCat("Non-existent src node: ", edge.src.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    const auto dst = node_name_to_index_.find(edge.dst.node->name());
    const bool valid_dst = dst != node_name_to_index_.end();

    if (!valid_dst) {
      const string error_message =
          absl::StrCat("Non-existent dst node: ", edge.dst.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    if (valid_dst && valid_src) {
      // TODO(benbarsdell): Check for failure.
      int src_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.src.node->name(), edge.src.type_attr));
      int dst_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.dst.node->name(), edge.dst.type_attr));
      fanins_[dst_node_type_idx].push_back(src_node_type_idx);
      fanouts_[src_node_type_idx].push_back(dst_node_type_idx);
    }
  }

  // 2. Add graph edges to the adjacency lists.
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    const NodeTypeId& node_type = node_type_attrs_.at(node_type_idx);
    auto input_ports =
        node_type_map.GetInputPorts(*node_type.node, node_type.type_attr);
    fanins_[node_type_idx].reserve(input_ports.size());
    for (int port : input_ports) {
      const string& input = node_type.node->input(port);
      TensorId tensor = ParseTensorName(input);
      const auto it = node_name_to_index_.find(tensor.node());
      const bool valid_input = it != node_name_to_index_.end();

      if (!valid_input) {
        const string error_message = absl::StrCat(
            "Non-existent input ", input, " in node ", node_type.node->name());
        if (skip_invalid_edges_) {
          VLOG(3) << "Skip error: " << error_message;
        } else {
          return errors::InvalidArgument(error_message);
        }
      }

      if (valid_input) {
        const int input_idx = it->second;
        const NodeDef& input_node = graph_->node(input_idx);
        TypeAttrId input_type_attr =
            node_type_map.GetOutputTypeAttr(input_node, tensor.index());
        const auto it2 = node_type_name_to_index_.find(
            NodeTypeKey(input_node.name(), input_type_attr));
        if (it2 == node_type_name_to_index_.end()) {
          if (!skip_invalid_edges_) {
            return errors::InvalidArgument("Did not find type attr ",
                                           input_type_attr.DebugString(),
                                           " in node ", input_node.name());
          }
          continue;
        }
        int input_node_type_idx = it2->second;
        fanins_[node_type_idx].push_back(input_node_type_idx);
        fanouts_[input_node_type_idx].push_back(node_type_idx);
      }
    }

    // Dedup the input list while it's still hot in cache.
    SortAndRemoveDuplicates(&fanins_[node_type_idx]);
  }

  // Dedup outputs for all the graph nodes.
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    SortAndRemoveDuplicates(&fanouts_[node_type_idx]);
  }

  return Status::OK();
}

Status GraphTypeTopologyView::InitializeFromGraph(
    const GraphDef& graph, const NodeTypeAttrMap& node_type_map) {
  return InitializeFromGraph(graph, node_type_map,
                             absl::Span<const NodeTypeIdEdge>());
}

bool GraphTypeTopologyView::HasNode(absl::string_view node_name,
                                    const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  return it != node_type_name_to_index_.end();
}

const NodeTypeId* GraphTypeTopologyView::GetNode(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  return it == node_type_name_to_index_.end()
             ? nullptr
             : &node_type_attrs_.at(it->second);
}

const NodeTypeId* GraphTypeTopologyView::GetNode(int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  DCHECK(node_idx >= 0 && node_idx < num_nodes_) << "node_idx is out of range";
  return &node_type_attrs_.at(node_idx);
}

const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  DCHECK(it != node_type_name_to_index_.end())
      << "Node doesn't exist in a graph";
  return it == node_type_name_to_index_.end() ? absl::nullopt
                                              : absl::make_optional(it->second);
}

const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    const NodeTypeId& node) const {
  return GetNodeIndex(node.node->name(), node.type_attr);
}

const absl::InlinedVector<int, 4>& GraphTypeTopologyView::GetFanin(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanins_[node_idx] : empty_fanin_;
}

const absl::InlinedVector<int, 2>& GraphTypeTopologyView::GetFanout(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanouts_[node_idx] : empty_fanout_;
}

enum class TypeTraversalDirection {
  kFollowInputs,
  kFollowOutputs,
  kFollowInputsAndOutputs,
};

// Encapsulate DFS callbacks that will be called during the graph traversal.
//
// If non-empty, the `pre_order` and `post_order` functors will be called on
// each reachable node (including the `from` nodes) in pre and post order. If
// loops are found, the `on_back_edge` functor will be called on the
// corresponding back edges. Moreover, the pre and post order will assume that
// these back edges will be cut.
struct DfsTypeCallbacks {
  DfsTypeCallbacks() = default;
  DfsTypeCallbacks(std::function<void(int)> pre, std::function<void(int)> post,
                   std::function<void(int, int)> back_edge)
      : pre_order(std::move(pre)),
        post_order(std::move(post)),
        on_back_edge(std::move(back_edge)) {}

  static DfsTypeCallbacks PreOrder(std::function<void(int)> pre) {
    return DfsTypeCallbacks(std::move(pre), nullptr, nullptr);
  }

  static DfsTypeCallbacks PostOrder(std::function<void(int)> post) {
    return DfsTypeCallbacks(nullptr, std::move(post), nullptr);
  }

  std::function<void(int)> pre_order;
  std::function<void(int)> post_order;
  std::function<void(int, int)> on_back_edge;
};

// Encapsulate DFS predicates for traversing the graph.
//
// The `enter` predicate decides if traversal should enter the node, and the
// `advance` predicate decides if the traversal should follow inputs/outputs
// from the node.
//
// If predicates are empty (default initialized), it's assumed that we can enter
// into any node and advance from any node respectively.
struct DfsTypePredicates {
  DfsTypePredicates() = default;
  DfsTypePredicates(std::function<bool(int)> enter,
                    std::function<bool(int)> advance)
      : enter(std::move(enter)), advance(std::move(advance)) {}

  static DfsTypePredicates Enter(std::function<bool(int)> enter) {
    return DfsTypePredicates(std::move(enter), nullptr);
  }

  static DfsTypePredicates Advance(std::function<bool(int)> advance) {
    return DfsTypePredicates(nullptr, std::move(advance));
  }

  std::function<bool(int)> enter;
  std::function<bool(int)> advance;
};

struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {}

  // Index of the node in the graph ∊ [0, num_nodes).
  int node;
  // `True` if visited all the input/output nodes (pushed all input/output nodes
  // to the stack).
  bool children_visited;
  // Index of the node in the graph, from which we entered the `node`.
  int src;
};

enum class NodeState { kNotVisited, kVisiting, kDone };

void DfsTypeTraversal(const GraphTypeTopologyView& graph_type_view,
                      const absl::Span<const NodeTypeId* const> from,
                      const TypeTraversalDirection direction,
                      const DfsTypePredicates& predicates,
                      const DfsTypeCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());

  for (const NodeTypeId* node : from) {
    const absl::optional<int> node_idx = graph_type_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value())
        << "Illegal start node: " << node->node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }

  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();

    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;

    // Skip nodes that we should not enter.
    if (predicates.enter && !predicates.enter(w.node)) {
      state = NodeState::kDone;
      continue;
    }

    // We've processed all the children of this node.
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(w.node);
      }
      continue;
    }

    // Loop detected.
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(w.src, w.node);
      }
      continue;
    }

    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.emplace_back(w.node, true, w.src);

    // Check if we can continue traversal from the current node.
    if (predicates.advance && !predicates.advance(w.node)) {
      continue;
    }

    // Now enqueue the fanin/fanout nodes.
    if (direction == TypeTraversalDirection::kFollowInputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanin : graph_type_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    }
    if (direction == TypeTraversalDirection::kFollowOutputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanout : graph_type_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}

DataTypeSet AllowedDataTypes(const OpDef::AttrDef& attr_def) {
  const auto& allowed_types = attr_def.allowed_values().list().type();
  if (allowed_types.empty()) {
    return AllTypes();
  }
  uint32 dtype_mask = 0;
  for (int dtype : allowed_types) {
    dtype_mask |= 1u << dtype;
  }
  return DataTypeSet(dtype_mask);
}

DataTypeSet AllowedDataTypes(const OpDef& op_def, const TypeAttrId& t_attr_id) {
  if (t_attr_id.attr_name.empty()) {
    return ToSet(t_attr_id.fixed_type);
  }
  const OpDef::AttrDef* attr_def = FindAttr(t_attr_id.attr_name, op_def);
  CHECK(attr_def);  // Crash Ok
  return AllowedDataTypes(*attr_def);
}

NodeDef BuildCastNode(const MutableGraphView::OutputPort& src, bool to_fp16,
                      const string& device) {
  const char* cast_string = to_fp16 ? kCastToFp16 : kCastToFp32;
  string name = strings::StrCat(src.node->name(), "-", src.port_id, "-",
                                cast_string, "-", kSuffix);
  NodeDef node;
  node.set_name(name);
  node.set_op("Cast");
  node.set_device(device);
  node.add_input(strings::StrCat(src.node->name(), ":", src.port_id));
  (*node.mutable_attr())["SrcT"].set_type(to_fp16 ? DT_FLOAT : DT_HALF);
  (*node.mutable_attr())["DstT"].set_type(to_fp16 ? DT_HALF : DT_FLOAT);
  (*node.mutable_attr())["Truncate"].set_b(false);
  return node;
}

Status ValidateLists(const gtl::FlatSet<string>& white_list,
                     const gtl::FlatSet<string>& black_list,
                     const gtl::FlatSet<string>& gray_list,
                     const gtl::FlatSet<string>& clear_list) {
  std::vector<gtl::FlatSet<string>> lists{white_list, black_list, gray_list,
                                          clear_list};
  std::multiset<string> counts;
  for (auto list : lists) {
    counts.insert(list.begin(), list.end());
  }
  bool duplicates = false;
  for (auto s : counts) {
    if (counts.count(s) > 1) {
      duplicates = true;
      LOG(ERROR) << "Op present in multiple lists: " << s;
    }
  }
  if (duplicates) {
    return errors::InvalidArgument("Op lists have conflicting entries");
  } else {
    return Status::OK();
  }
}

bool HasInputOrOutputRefs(const NodeDef& node) {
  const OpDef* op_def;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return true;
  }
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return true;
    }
  }
  for (const auto& output : op_def->output_arg()) {
    if (output.is_ref()) {
      return true;
    }
  }
  return false;
}

// See TF issue 25977 for no-FP16 on SCEWL
bool CanForceFP16(const NodeDef& node) {
  return node.op() != "Const" && node.op() != "SoftmaxCrossEntropyWithLogits" &&
         !IsStateful(node) && !HasInputOrOutputRefs(node);
}

int GetCudaVersion(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      const auto& device_env = device_properties.environment();
      auto it = device_env.find("cuda");
      if (it != device_env.end()) {
        string cuda_version_str = it->second;
        return std::stoi(cuda_version_str);
      }
    }
  }
  return 0;
}

int GetCudnnVersion(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      const auto& device_env = device_properties.environment();
      auto it = device_env.find("cudnn");
      if (it != device_env.end()) {
        string cudnn_version_str = it->second;
        return std::stoi(cudnn_version_str);
      }
    }
  }
  return 0;
}

class AutoMixedPrecisionImpl {
 public:
  AutoMixedPrecisionImpl(Cluster* cluster,
                         const std::unordered_set<string>& nodes_to_preserve,
                         GraphDef* graph, string id)
      : virtual_placer_(cluster->GetDevices()),
        nodes_to_preserve_(nodes_to_preserve),
        graph_(graph),
        id_(id),
        graph_view_(graph),
        cuda_version_(GetCudaVersion(*cluster)),
        cudnn_version_(GetCudnnVersion(*cluster)) {}

  Status Optimize();

 private:
  typedef absl::flat_hash_set<NodeTypeId> NodeTypeIdSet;
  // Maps data structure object ops (e.g., StackV2) to the sets of nodes that
  // write (e.g., StackPushV2) and read (e.g., StackPopV2) from them.
  typedef absl::flat_hash_map<NodeTypeId,
                              std::pair<NodeTypeIdSet, NodeTypeIdSet>>
      DataStructureOpsMap;

  Status PrintDebugLogs(bool preop, size_t timestamp);
  void LogSkippedNode(const NodeDef& node) const;
  bool MustPreserve(const NodeDef& node) const;
  bool IsOnGPU(const NodeDef& node) const;
  bool IsOnSuitableGPUArch(const NodeDef& node) const;
  bool ShouldProcess(const NodeDef& node) const;
  bool NodeHasFP16KernelForTypeAttr(const NodeDef& node, TypeAttrId taid) const;
  bool NodeImplicitlyReadsNonResourceVariable(const NodeDef& node) const;
  void ConvertBatchNormOpsToV2();
  bool SupportsFloat16(const NodeTypeId& node_type) const;
  const NodeDef* GetTailOfChain(
      const NodeDef& node, const absl::flat_hash_set<string>& match_ops) const;
  Status AddDataStructureOpsToMap(
      const absl::flat_hash_set<string>& data_structure_ops,
      TypeAttrId data_structure_type_attr,
      const absl::flat_hash_map<string, TypeAttrId>& write_ops,
      const absl::flat_hash_map<string, TypeAttrId>& read_ops,
      DataStructureOpsMap* object_clients_map) const;
  void AddWhitelistOps(absl::flat_hash_set<int>* white_set) const;
  void PropagateBlackFwdThroughClearAndGray(
      absl::flat_hash_set<int>* black_set) const;
  void ForceColorMatchBetweenDataStructureOps(
      const DataStructureOpsMap& object_clients_map,
      absl::flat_hash_set<int>* white_set,
      absl::flat_hash_set<int>* black_set) const;
  void AddClearAndGrayToWhiteIfBetweenWhite(
      const absl::flat_hash_set<int>& black_set,
      absl::flat_hash_set<int>* white_set) const;
  void PropagateWhiteThroughClear(const absl::flat_hash_set<int>& black_set,
                                  absl::flat_hash_set<int>* white_set) const;
  Status ForceColorMatchOnRecurrentEdges(
      absl::flat_hash_set<int>* white_set) const;
  void MakeCastsWhiteIfAllOutputsWhite(
      absl::flat_hash_set<int>* white_set) const;
  Status ChangeTypeAttrsAndAddCasts(const absl::flat_hash_set<int>& white_set);

  VirtualPlacer virtual_placer_;
  std::unordered_set<string> nodes_to_preserve_;
  GraphDef* graph_;
  string id_;
  MutableGraphView graph_view_;
  int cuda_version_;
  int cudnn_version_;
  NodeTypeAttrMap node_type_map_;
  GraphTypeTopologyView graph_type_view_;
  bool force_all_fp16_;
  gtl::FlatSet<string> fp16_whitelist_;
  gtl::FlatSet<string> fp16_blacklist_;
  gtl::FlatSet<string> fp16_graylist_;
  gtl::FlatSet<string> fp16_clearlist_;
  absl::flat_hash_set<const NodeDef*> should_process_nodes_;
};

bool AutoMixedPrecisionImpl::NodeHasFP16KernelForTypeAttr(
    const NodeDef& node, TypeAttrId taid) const {
  NodeDef node_copy(node);
  if (node.device().empty()) {
    string device_name = virtual_placer_.get_canonical_device_name(node);
    node_copy.set_device(device_name);
  }
  if (!SetDataType(&node_copy, taid, DataType::DT_HALF)) {
    return false;
  }
  return IsKernelRegisteredForNode(node_copy).ok();
}

Status AutoMixedPrecisionImpl::PrintDebugLogs(bool preop, size_t timestamp) {
  string prepend_path;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH", "", &prepend_path));
  if (prepend_path.empty()) return Status::OK();

  string suffix =
      strings::StrCat("_", preop ? "preop" : kSuffix, "_", id_, "_", timestamp);

  string fname =
      io::JoinPath(prepend_path, strings::StrCat("graphdef", suffix, ".pb"));
  std::fstream f;
  f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
  f << graph_->SerializeAsString();
  f.close();
  LOG(INFO) << "Saved " << (preop ? "pre-optimization" : "post-optimization")
            << " graph as binary to " << fname;

  fname = io::JoinPath(prepend_path,
                       strings::StrCat("graphdef", suffix, ".pb.txt"));
  f.open(fname.c_str(), std::fstream::out);
  f << graph_->DebugString();
  f.close();
  LOG(INFO) << "Saved " << (preop ? "pre-optimization" : "post-optimization")
            << " graph as text to " << fname;

  if (!preop) {
    fname = io::JoinPath(prepend_path,
                         strings::StrCat("paintbuckets", suffix, ".txt"));
    f.open(fname.c_str(), std::fstream::out);
    f << "WhiteList:\n";
    for (auto x :
         AutoMixedPrecisionLists::WhiteList(cuda_version_, cudnn_version_)) {
      f << x << "\n";
    }
    f << "\nBlackList:\n";
    for (auto x : AutoMixedPrecisionLists::BlackList()) {
      f << x << "\n";
    }
    f << "\nGrayList:\n";
    for (auto x : AutoMixedPrecisionLists::GrayList()) {
      f << x << "\n";
    }
    f << "\nClearList:\n";
    for (auto x : AutoMixedPrecisionLists::ClearList()) {
      f << x << "\n";
    }
    f.close();
    LOG(INFO) << "Saved paint bucket info to " << fname;
  }
  return Status::OK();
}

void AutoMixedPrecisionImpl::LogSkippedNode(const NodeDef& node) const {
  VLOG(2) << "Skipping " << node.op() << " node " << node.name()
          << " because it "
          << (MustPreserve(node)
                  ? "must be preserved"
                  : "is not on the GPU, or the GPU arch is not suitable");
}

bool AutoMixedPrecisionImpl::MustPreserve(const NodeDef& node) const {
  return nodes_to_preserve_.count(node.name());
}

bool AutoMixedPrecisionImpl::IsOnGPU(const NodeDef& node) const {
  string device_name;
  if (node.device().empty()) {
    device_name = virtual_placer_.get_canonical_device_name(node);
  } else {
    device_name = node.device();
  }
  string device;
  string not_used;
  if (DeviceNameUtils::SplitDeviceName(device_name, &not_used, &device) &&
      absl::StrContains(absl::AsciiStrToLower(device),
                        absl::AsciiStrToLower(DEVICE_GPU))) {
    return true;
  }
  return false;
}

// Returns the GPU architecture (compute capability) as a (major, minor) pair.
std::pair<int, int> GetDeviceGPUArch(
    const DeviceProperties& device_properties) {
  if (device_properties.type() != "GPU") return {0, 0};
  string arch_str = device_properties.environment().at("architecture");
  std::vector<string> split_arch_str = str_util::Split(arch_str, '.');
  if (split_arch_str.empty()) {
    return {0, 0};
  }

  int major, minor;
  if (!strings::safe_strto32(split_arch_str[0], &major)) {
    return {0, 0};
  }

  if (split_arch_str.size() > 1) {
    if (strings::safe_strto32(split_arch_str[1], &minor)) {
      return {major, minor};
    } else {
      return {0, 0};
    }
  } else {
    return {major, 0};
  }
}

bool AutoMixedPrecisionImpl::IsOnSuitableGPUArch(const NodeDef& node) const {
  return GetDeviceGPUArch(virtual_placer_.get_device(node)) >= kMinGPUArch;
}

bool AutoMixedPrecisionImpl::ShouldProcess(const NodeDef& node) const {
  return should_process_nodes_.count(&node);
}

bool IsFloat32(const NodeTypeId& node_type) {
  return GetDataType(*node_type.node, node_type.type_attr) ==
         DataType::DT_FLOAT;
}

bool AutoMixedPrecisionImpl::SupportsFloat16(
    const NodeTypeId& node_type) const {
  const OpDef* op_def;
  Status status =
      OpRegistry::Global()->LookUpOpDef(node_type.node->op(), &op_def);
  if (!status.ok()) return false;
  return AllowedDataTypes(*op_def, node_type.type_attr)
             .Contains(DataType::DT_HALF) &&
         NodeHasFP16KernelForTypeAttr(*node_type.node, node_type.type_attr);
}

// TODO(mconley): Make this change the node's name (to aid debugging). Need to
// make sure that doing this won't break anything.
void AutoMixedPrecisionImpl::ConvertBatchNormOpsToV2() {
  for (int node_idx = 0; node_idx < graph_->node_size(); ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    if (!ShouldProcess(*node)) continue;
    bool changed = false;
    if (node->op() == "FusedBatchNorm") {
      VLOG(2) << "Changing op of " << node->op() << " node " << node->name()
              << " to FusedBatchNormV2";
      node->set_op("FusedBatchNormV2");
      changed = true;
    } else if (node->op() == "FusedBatchNormGrad") {
      VLOG(2) << "Changing op of " << node->op() << " node " << node->name()
              << " to FusedBatchNormGradV2";
      node->set_op("FusedBatchNormGradV2");
      changed = true;
    }
    if (changed) {
      (*node->mutable_attr())["U"].set_type(DT_FLOAT);
    }
  }
}

// A helper function to decide whether to ignore the effect on performance when
// rewriting the graph. This can be useful for testing the numerical effects of
// reduced precision on systems that have poor mixed precision performance.
bool ShouldIgnorePerformance() {
  static bool is_enabled = [] {
    bool ret = false;
    TF_CHECK_OK(ReadBoolFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE",
        /*default_val=*/false, &ret));
    return ret;
  }();
  return is_enabled;
}

Status AutoMixedPrecisionImpl::Optimize() {
  string optimization_level;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "", &optimization_level));
  optimization_level = absl::AsciiStrToUpper(optimization_level);
  force_all_fp16_ = optimization_level == "UNSAFE_FORCE_ALL";

  fp16_whitelist_ =
      AutoMixedPrecisionLists::WhiteList(cuda_version_, cudnn_version_);
  fp16_blacklist_ = AutoMixedPrecisionLists::BlackList();
  fp16_graylist_ = AutoMixedPrecisionLists::GrayList();
  fp16_clearlist_ = AutoMixedPrecisionLists::ClearList();
  TF_RETURN_IF_ERROR(ValidateLists(fp16_whitelist_, fp16_blacklist_,
                                   fp16_graylist_, fp16_clearlist_));

  size_t timestamp = Env::Default()->NowMicros() / 1000;
  TF_RETURN_IF_ERROR(PrintDebugLogs(/* preop = */ true, timestamp));

  VLOG(2) << "Identifying nodes that should be processed";
  for (const NodeDef& node : graph_->node()) {
    if (!MustPreserve(node) && IsOnGPU(node) &&
        (ShouldIgnorePerformance() || IsOnSuitableGPUArch(node))) {
      should_process_nodes_.insert(&node);
    } else {
      LogSkippedNode(node);
    }
  }

  VLOG(2) << "Converting FusedBatchNorm* ops to V2";
  ConvertBatchNormOpsToV2();

  VLOG(2) << "Building node type map for graph";
  TF_RETURN_IF_ERROR(node_type_map_.Init(*graph_));

  // Note: If an op is added to this list that has a data type attribute, it
  // should also be added to the AddDataStructureOpsToMap call below (and to the
  // clearlist if it involves data flow).
  // TODO(benbarsdell): Add support for TensorListPushBackBatch and
  // TensorListConcatLists. They require special handling because they connect
  // multiple list objects together. Currently if they appear in the graph then
  // we have no choice but to disallow changing any tensor list ops, as
  // otherwise we risk breaking the graph if some are changed and some are not
  // (within a connected cluster of tensor list nodes).
  const gtl::FlatSet<string> supported_list_ops = {
      "EmptyTensorList",
      "TensorListSplit",
      "TensorListFromTensor",
      "TensorListReserve",
      "TensorListScatter",
      "TensorListScatterV2",
      "TensorListPushBack",
      "TensorListSetItem",
      "TensorListScatterIntoExistingList",
      "TensorListPopBack",
      "TensorListStack",
      "TensorListConcat",
      "TensorListConcatV2",
      "TensorListGetItem",
      "TensorListGather",
      "TensorListLength",
      "TensorListElementShape",
      "TensorListResize"};

  bool can_change_tensor_list_ops = true;
  for (const NodeDef& node : graph_->node()) {
    if (absl::StartsWith(node.op(), "TensorList") &&
        !supported_list_ops.count(node.op())) {
      LOG(WARNING) << "Unsupported " << node.op() << " node found in graph ("
                   << node.name()
                   << "), tensor list ops will not be converted.";
      can_change_tensor_list_ops = false;
      break;
    }
  }

  DataStructureOpsMap object_clients_map;
  if (can_change_tensor_list_ops) {
    VLOG(2) << "Identifying TensorList* nodes";
    TF_RETURN_IF_ERROR(AddDataStructureOpsToMap(
        {"EmptyTensorList", "TensorListSplit", "TensorListFromTensor",
         "TensorListReserve", "TensorListScatter", "TensorListScatterV2"},
        TypeAttrId("element_dtype"),
        {{"TensorListPushBack", TypeAttrId("element_dtype")},
         {"TensorListSetItem", TypeAttrId("element_dtype")},
         {"TensorListScatterIntoExistingList", TypeAttrId("element_dtype")}},
        {{"TensorListPopBack", TypeAttrId("element_dtype")},
         {"TensorListStack", TypeAttrId("element_dtype")},
         {"TensorListConcat", TypeAttrId("element_dtype")},
         {"TensorListConcatV2", TypeAttrId("element_dtype")},
         {"TensorListGetItem", TypeAttrId("element_dtype")},
         {"TensorListGather", TypeAttrId("element_dtype")}},
        &object_clients_map));
  } else {
    for (const string& list_op : supported_list_ops) {
      fp16_whitelist_.erase(list_op);
      fp16_graylist_.erase(list_op);
      fp16_clearlist_.erase(list_op);
    }
  }

  // Create ephemeral edges between writers and readers of data structure ops.
  std::vector<NodeTypeIdEdge> ephemeral_edges;
  for (const auto& object_clients : object_clients_map) {
    const auto& client_nodes = object_clients.second;
    for (const NodeTypeId& write_node_type : client_nodes.first) {
      for (const NodeTypeId& read_node_type : client_nodes.second) {
        ephemeral_edges.emplace_back(write_node_type, read_node_type);
      }
    }
    const NodeTypeId& object_node_type = object_clients.first;
    // These object types also act as writers because they initialize the object
    // from an input tensor.
    if (object_node_type.node->op() == "TensorListSplit" ||
        object_node_type.node->op() == "TensorListFromTensor" ||
        object_node_type.node->op() == "TensorListScatter" ||
        object_node_type.node->op() == "TensorListScatterV2") {
      for (const NodeTypeId& read_node_type : client_nodes.second) {
        ephemeral_edges.emplace_back(object_node_type, read_node_type);
      }
    }
  }

  VLOG(2) << "Constructing graph type attribute topology view";
  TF_RETURN_IF_ERROR(graph_type_view_.InitializeFromGraph(
      *graph_, node_type_map_, ephemeral_edges));

  // The goal here is to change performance-critical ops to fp16, and to do so
  // with the minimal number of casts, subject to the constraint that the
  // model's convergence is not affected. This is achieved by first identifying
  // which nodes should be changed to fp16 and then inserting casts at the
  // boundaries between fp16/non-fp16 nodes.

  // The algorithm for deciding which nodes to change to fp16 is as follows:
  // 1) Add all performance-critical ops (aka "whitelist" ops) to the white_set.
  //    This is done under the assumption that whitelist ops are always
  //    numerically-safe in fp16 and that they are the most important ops for
  //    improving performance.
  // 2) Add nodes to the black_set iff they are numerically-dangerous (aka
  //    "blacklist" ops) or they are on a forward path from a blacklist node to
  //    a black/gray node (including the node at the end of the path) through
  //    non-numerically-dangerous ops (aka "greylist" and "clearlist" ops).
  //    This is done to prevent numerically-dangerous ops and their downstream
  //    effects from being changed to fp16, which would risk breaking the
  //    numerical accuracy of the model.
  // 3) For all remaining nodes that are not considered dangerous (greylist
  //    and clearlist ops), find those that are between (i.e., both upstream
  //    and downstream of) white nodes, and add them to the white_set.
  //    This is done to avoid unnecessary casts between whitelist ops.
  // 4) For all remaining clearlist nodes, add them to the white_set if they are
  //    connected to a node in the white_set via other clearlist nodes.
  //    This is done to increase the number of ops in the white_set without
  //    affecting numerical stability.

  absl::flat_hash_set<int> white_set;
  VLOG(2) << "Beginning pass 1 to add whitelist ops";
  AddWhitelistOps(&white_set);
  VLOG(2) << "Finished pass 1";

  if (white_set.empty()) {
    LOG(INFO) << "No whitelist ops found, nothing to do";
    return Status::OK();
  }

  absl::flat_hash_set<int> black_set;
  VLOG(2) << "Beginning pass 2 to propagate black forwards from blacklist ops "
             "through clear/graylist ops";
  PropagateBlackFwdThroughClearAndGray(&black_set);
  VLOG(2) << "Finished pass 2";

  VLOG(2) << "Forcing color match between data structure ops";
  ForceColorMatchBetweenDataStructureOps(object_clients_map, &white_set,
                                         &black_set);

  VLOG(2) << "Beginning pass 3 to set clear and gray nodes to white if they "
             "are between white ops";
  AddClearAndGrayToWhiteIfBetweenWhite(black_set, &white_set);
  VLOG(2) << "Finished pass 3";

  VLOG(2) << "Beginning pass 4 to propagate white from white nodes through "
             "clearlist ops";
  PropagateWhiteThroughClear(black_set, &white_set);
  VLOG(2) << "Finished pass 4";

  VLOG(2) << "Forcing color match between data structure ops";
  ForceColorMatchBetweenDataStructureOps(object_clients_map, &white_set,
                                         &black_set);

  VLOG(2) << "Forcing color match on loop edges";
  TF_RETURN_IF_ERROR(ForceColorMatchOnRecurrentEdges(&white_set));

  VLOG(2) << "Finding existing casts that can be made white";
  MakeCastsWhiteIfAllOutputsWhite(&white_set);

  VLOG(2) << "Beginning final pass to change type attributes and insert Cast "
             "ops at paint boundaries";
  TF_RETURN_IF_ERROR(ChangeTypeAttrsAndAddCasts(white_set));
  VLOG(2) << "Finished final pass";

  TF_RETURN_IF_ERROR(PrintDebugLogs(/* preop = */ false, timestamp));

  return Status::OK();
}

// Finds data structure object ops (e.g., StackV2) and the sets of nodes that
// write (e.g., StackPushV2) and read (e.g., StackPopV2) from them.
Status AutoMixedPrecisionImpl::AddDataStructureOpsToMap(
    const absl::flat_hash_set<string>& data_structure_ops,
    TypeAttrId data_structure_type_attr,
    const absl::flat_hash_map<string, TypeAttrId>& write_ops,
    const absl::flat_hash_map<string, TypeAttrId>& read_ops,
    DataStructureOpsMap* object_clients_map) const {
  for (const NodeDef& node : graph_->node()) {
    const auto write_iter = write_ops.find(node.op());
    const auto read_iter = read_ops.find(node.op());
    bool is_writer = write_iter != write_ops.end();
    bool is_reader = read_iter != read_ops.end();
    if (is_writer || is_reader) {
      const NodeDef* object_node = GetTailOfChain(node, data_structure_ops);
      if (!object_node) {
        return errors::FailedPrecondition(
            "No data structure op found upstream of ", node.op(), " node ",
            node.name());
      }
      NodeTypeId object_node_type(object_node, data_structure_type_attr);
      TypeAttrId type_attr = is_writer ? write_iter->second : read_iter->second;
      NodeTypeId node_type(&node, type_attr);
      auto* value = &(*object_clients_map)[object_node_type];
      auto* node_set = is_writer ? &value->first : &value->second;
      node_set->insert(node_type);
    }
  }
  return Status::OK();
}

void AutoMixedPrecisionImpl::AddWhitelistOps(
    absl::flat_hash_set<int>* white_set) const {
  // Add whitelisted ops to white_set.
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node)) continue;
    bool force_white = force_all_fp16_ && CanForceFP16(*root.node);
    if (fp16_whitelist_.count(root.node->op()) || force_white) {
      bool inserted = white_set->insert(root_idx).second;
      if (VLOG_IS_ON(2) && inserted) {
        VLOG(2) << "Painting type " << root.type_attr.DebugString()
                << " of node " << root.node->name() << " WHITE because its op "
                << root.node->op() << " is on the whitelist";
      }
    }
  }
}

// Adds nodes to black_set iff they are on the blacklist or they are on a
// forward path from a blacklist node to a black/gray node (including the node
// at the end of the path) through clear and gray nodes.
// E.g., black -> gray -> clear -> gray -> clear -> white -> gray
// becomes: black -> black -> black -> black -> clear -> white -> gray.
void AutoMixedPrecisionImpl::PropagateBlackFwdThroughClearAndGray(
    absl::flat_hash_set<int>* black_set) const {
  if (force_all_fp16_) return;

  // Find clear nodes that are upstream of black or gray.
  absl::flat_hash_set<int> upstream_of_black_or_gray_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!(fp16_blacklist_.count(root.node->op()) ||
          fp16_graylist_.count(root.node->op()))) {
      continue;
    }
    DfsTypeTraversal(graph_type_view_, {&root},
                     TypeTraversalDirection::kFollowInputs,
                     DfsTypePredicates::Enter([&](int idx) -> bool {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       return idx == root_idx ||
                              (!upstream_of_black_or_gray_set.count(idx) &&
                               fp16_clearlist_.count(item.node->op()));
                     }),
                     DfsTypeCallbacks::PreOrder([&](int idx) {
                       upstream_of_black_or_gray_set.insert(idx);
                     }));
  }

  // Propagate black forward through nodes in upstream_of_black_or_gray_set.
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (black_set->count(root_idx) || !fp16_blacklist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!black_set->count(idx) &&
                                     upstream_of_black_or_gray_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          bool inserted = black_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " BLACK";
          }
        }));
  }
}

void AutoMixedPrecisionImpl::AddClearAndGrayToWhiteIfBetweenWhite(
    const absl::flat_hash_set<int>& black_set,
    absl::flat_hash_set<int>* white_set) const {
  // Find clear/graylist ops that are downstream of white ops.
  absl::flat_hash_set<int> downstream_of_white_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || !fp16_whitelist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!downstream_of_white_set.count(idx) &&
                  !fp16_whitelist_.count(item.node->op()) &&
                  !black_set.count(idx) && ShouldProcess(*item.node) &&
                  // TODO(benbarsdell): Consider allowing propagation through
                  // ops that are already float16 in order to reduce the number
                  // of casts.
                  IsFloat32(item) && SupportsFloat16(item) &&
                  (fp16_clearlist_.count(item.node->op()) ||
                   fp16_graylist_.count(item.node->op())));
        }),
        DfsTypeCallbacks::PreOrder(
            [&](int idx) { downstream_of_white_set.insert(idx); }));
  }

  // Set nodes that are both downstream and upstream of white ops to white.
  absl::flat_hash_set<int> upstream_of_white_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || upstream_of_white_set.count(root_idx) ||
        !fp16_whitelist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!upstream_of_white_set.count(idx) &&
                                     downstream_of_white_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          upstream_of_white_set.insert(idx);
          bool inserted = white_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " WHITE";
          }
        }));
  }
}

void AutoMixedPrecisionImpl::PropagateWhiteThroughClear(
    const absl::flat_hash_set<int>& black_set,
    absl::flat_hash_set<int>* white_set) const {
  // Propagate white from white nodes through clearlist ops.
  absl::flat_hash_set<int> clear_prop_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || clear_prop_set.count(root_idx) ||
        !white_set->count(root_idx)) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root},
        TypeTraversalDirection::kFollowInputsAndOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!white_set->count(idx) && !black_set.count(idx) &&
                  ShouldProcess(*item.node) && IsFloat32(item) &&
                  SupportsFloat16(item) &&
                  (fp16_clearlist_.count(item.node->op())) &&
                  // We don't propagate (backwards) through nodes that read
                  // Variables because it can break the behavior of TensorBoard
                  // visualization and/or (in the case of Enter nodes) the model
                  // itself. This is only a problem for non-resource variables.
                  !NodeImplicitlyReadsNonResourceVariable(*item.node));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          clear_prop_set.insert(idx);
          bool inserted = white_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " WHITE";
          }
        }));
  }
}

// Forces NextIteration nodes and their output Merge node(s) to have the same
// color. Specifically, it removes them all from white_set if any of the Merge
// nodes is not in white_set, otherwise it adds the NextIteration node to
// white_set.
Status AutoMixedPrecisionImpl::ForceColorMatchOnRecurrentEdges(
    absl::flat_hash_set<int>* white_set) const {
  for (const NodeDef& node : graph_->node()) {
    if (node.op() == "NextIteration") {
      GraphView::OutputPort output_port(&node, 0);
      const auto& fanout = graph_view_.GetFanout(output_port);
      std::vector<int> merge_idxs;
      merge_idxs.reserve(fanout.size());
      bool any_merge_is_not_white = false;
      for (const auto& output : fanout) {
        const NodeDef& merge_node = *output.node;
        if (merge_node.op() != "Merge") {
          return errors::FailedPrecondition(
              "Expected Merge node after NextIteration, got ", merge_node.op());
        }
        const absl::optional<int> maybe_merge_idx =
            graph_type_view_.GetNodeIndex(merge_node.name(), TypeAttrId("T"));
        if (!maybe_merge_idx.has_value()) {
          return errors::Internal("Type attribute T of Merge node ",
                                  merge_node.name(),
                                  " not found in graph view");
        }
        int merge_idx = maybe_merge_idx.value();
        merge_idxs.push_back(merge_idx);
        any_merge_is_not_white =
            any_merge_is_not_white || !white_set->count(merge_idx);
      }
      const absl::optional<int> maybe_nextiter_idx =
          graph_type_view_.GetNodeIndex(node.name(), TypeAttrId("T"));
      if (!maybe_nextiter_idx.has_value()) {
        return errors::Internal("Type attribute T of NextIteration node ",
                                node.name(), " not found in graph view");
      }
      int nextiter_idx = maybe_nextiter_idx.value();
      if (any_merge_is_not_white) {
        for (int merge_idx : merge_idxs) {
          if (white_set->erase(merge_idx)) {
            VLOG(2) << "Painting type T of Merge node "
                    << graph_type_view_.GetNode(merge_idx)->node->name()
                    << " BLACK to match the color of its sibling Merge nodes "
                       "with common NextIteration node "
                    << node.name();
          }
        }
        if (white_set->erase(nextiter_idx)) {
          VLOG(2) << "Painting type T of NextIteration node " << node.name()
                  << " BLACK to match the color of its output Merge node(s)";
        }
      } else {
        if (white_set->insert(nextiter_idx).second) {
          VLOG(2) << "Painting type T of NextIteration node " << node.name()
                  << " WHITE to match the color of its output Merge node(s)";
        }
      }
    }
  }
  return Status::OK();
}

// Returns the last node in the simple chain starting at node and traversing
// backwards through the input(0) edge from each node until one with a matching
// op is found, or nullptr if no matching node is found.
const NodeDef* AutoMixedPrecisionImpl::GetTailOfChain(
    const NodeDef& node, const absl::flat_hash_set<string>& match_ops) const {
  const NodeDef* node_ptr = &node;
  do {
    GraphView::InputPort node_input(node_ptr, 0);
    MutableGraphView::OutputPort prev_output =
        graph_view_.GetRegularFanin(node_input);
    node_ptr = prev_output.node;
  } while (node_ptr && !match_ops.count(node_ptr->op()));
  return node_ptr;
}

// Ensures that data structure nodes (e.g., StackV2) and all of their associated
// client nodes (e.g., StackPushV2 and StackPopV2) are in the same color set.
void AutoMixedPrecisionImpl::ForceColorMatchBetweenDataStructureOps(
    const DataStructureOpsMap& object_clients_map,
    absl::flat_hash_set<int>* white_set,
    absl::flat_hash_set<int>* black_set) const {
  for (const auto& object_clients : object_clients_map) {
    const NodeTypeId& object_node_type = object_clients.first;
    const auto& client_nodes = object_clients.second;
    NodeTypeIdSet all_client_nodes = client_nodes.first;
    all_client_nodes.insert(client_nodes.second.begin(),
                            client_nodes.second.end());
    // The object node may be considered a client too (e.g.,
    // TensorListFromTensor).
    all_client_nodes.insert(object_node_type);
    bool any_black = false;
    bool any_white = false;
    for (const NodeTypeId& node_type : all_client_nodes) {
      const absl::optional<int> maybe_node_idx =
          graph_type_view_.GetNodeIndex(node_type);
      DCHECK(maybe_node_idx.has_value())
          << "Type attribute " << node_type.type_attr.DebugString()
          << " of node " << node_type.node->name()
          << " not found in graph view";
      int node_idx = maybe_node_idx.value();
      if (black_set->count(node_idx)) {
        any_black = true;
        break;
      } else if (white_set->count(node_idx)) {
        any_white = true;
      }
    }
    if (any_black || any_white) {
      for (const NodeTypeId& node_type : all_client_nodes) {
        VLOG(2) << "Painting type " << node_type.type_attr.DebugString()
                << " of " << node_type.node->op() << " node "
                << node_type.node->name() << " "
                << (any_black ? "BLACK" : "WHITE")
                << " because at least one of its siblings is "
                << (any_black ? "BLACK" : "WHITE");
        const absl::optional<int> maybe_node_idx =
            graph_type_view_.GetNodeIndex(node_type);
        DCHECK(maybe_node_idx.has_value())
            << "Type attribute " << node_type.type_attr.DebugString()
            << " of node " << node_type.node->name()
            << " not found in graph view";
        int node_idx = maybe_node_idx.value();
        if (any_black) {
          white_set->erase(node_idx);
          black_set->insert(node_idx);
        } else {
          white_set->insert(node_idx);
        }
      }
    }
  }
}

bool AutoMixedPrecisionImpl::NodeImplicitlyReadsNonResourceVariable(
    const NodeDef& node) const {
  if (node.op() == "Identity" || node.op() == "Enter") {
    GraphView::InputPort node_input(&node, 0);
    MutableGraphView::OutputPort prev_output =
        graph_view_.GetRegularFanin(node_input);
    const NodeDef* input = prev_output.node;
    if (input && ((node.op() == "Identity" && (input->op() == "Variable" ||
                                               input->op() == "VariableV2")) ||
                  (node.op() == "Enter" &&
                   NodeImplicitlyReadsNonResourceVariable(*input)))) {
      return true;
    }
  }
  return false;
}

// This adds existing Cast nodes to white_set if all of their outputs are white,
// avoiding the need to add a new Cast node after an existing Cast.
void AutoMixedPrecisionImpl::MakeCastsWhiteIfAllOutputsWhite(
    absl::flat_hash_set<int>* white_set) const {
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    NodeTypeId node_type(node, TypeAttrId("DstT"));
    if (node->op() != "Cast" || !IsFloat32(node_type)) {
      continue;
    }
    bool all_fanouts_white = true;
    MutableGraphView::OutputPort src(node, 0);
    const auto& fanout = graph_view_.GetFanout(src);
    for (const MutableGraphView::InputPort& dst : fanout) {
      TypeAttrId dst_type_attr =
          node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
      const absl::optional<int> maybe_dst_type_idx =
          graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
      DCHECK(maybe_dst_type_idx.has_value())
          << "Type attribute " << dst_type_attr.DebugString() << " of node "
          << dst.node->name() << " not found in graph view";
      int dst_type_idx = maybe_dst_type_idx.value();
      bool dst_is_white = white_set->count(dst_type_idx);
      if (!dst_is_white) {
        all_fanouts_white = false;
        break;
      }
    }
    if (!fanout.empty() && all_fanouts_white) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node_type);
      DCHECK(maybe_node_type_idx.has_value())
          << "Type attribute " << node_type.type_attr.DebugString()
          << " of node " << node_type.node->name()
          << " not found in graph view";
      int node_type_idx = maybe_node_type_idx.value();
      white_set->insert(node_type_idx);
    }
  }
}

// Changes all white-painted type attributes to DT_HALF, and inserts Cast nodes
// at node outputs for all edges that connect white-painted <->
// non-white-painted type attributes.
Status AutoMixedPrecisionImpl::ChangeTypeAttrsAndAddCasts(
    const absl::flat_hash_set<int>& white_set) {
  int num_nodes_changed = 0;
  int num_nonvar_casts_to_fp16 = 0;
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(*node)) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node->name(), type_attr);
      if (!maybe_node_type_idx.has_value()) {
        return errors::Internal("Type attribute ", type_attr.DebugString(),
                                " of ", node->op(), " node ", node->name(),
                                " not found in graph view");
      }
      int node_type_idx = maybe_node_type_idx.value();
      if (!IsFloat32(*graph_type_view_.GetNode(node_type_idx))) continue;
      bool src_is_white = white_set.count(node_type_idx);
      if (src_is_white) {
        VLOG(1) << "Changing type " << type_attr.DebugString() << " of "
                << node->op() << " node " << node->name() << " to DT_HALF";
        if (!SetDataType(node, type_attr, DT_HALF)) {
          return errors::Internal("Failed to set type attribute");
        }
        ++num_nodes_changed;
      }
      for (int output_port : node_type_map_.GetOutputPorts(*node, type_attr)) {
        MutableGraphView::OutputPort src(node, output_port);
        NodeDef* added_cast_node = nullptr;
        // Note: This is copied so that edges can be modified inside the loop.
        auto fanout = graph_view_.GetFanout(src);
        for (const MutableGraphView::InputPort& dst : fanout) {
          TypeAttrId dst_type_attr =
              node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
          const absl::optional<int> maybe_dst_type_idx =
              graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
          if (!maybe_dst_type_idx.has_value()) {
            return errors::Internal("Type attribute ",
                                    dst_type_attr.DebugString(), " of ",
                                    dst.node->op(), " node ", dst.node->name(),
                                    " not found in graph view");
          }
          int dst_type_idx = maybe_dst_type_idx.value();
          bool dst_is_white = white_set.count(dst_type_idx);
          if (src_is_white != dst_is_white) {
            if (!added_cast_node) {
              bool to_fp16 = dst_is_white;
              VLOG(1) << "Inserting cast to "
                      << (to_fp16 ? "DT_HALF" : "DT_FLOAT") << " at "
                      << src.node->op() << " " << src.node->name() << ":"
                      << src.port_id;
              added_cast_node = graph_view_.AddNode(
                  BuildCastNode(src, to_fp16, src.node->device()));
              if (to_fp16 && !IsConstant(*node) && !IsVariable(*node) &&
                  !NodeImplicitlyReadsNonResourceVariable(*node)) {
                ++num_nonvar_casts_to_fp16;
              }
            }
            TF_RETURN_IF_ERROR(graph_view_.UpdateRegularFaninByPort(
                dst.node->name(), dst.port_id, {added_cast_node->name(), 0}));
          }
        }
      }
    }
  }
  LOG(INFO) << "Converted " << num_nodes_changed << "/" << num_nodes_preop
            << " nodes to float16 precision using " << num_nonvar_casts_to_fp16
            << " cast(s) to float16 (excluding Const and Variable casts)";
  return Status::OK();
}

int GetNumGPUs(const Cluster& cluster,
               const std::pair<int, int>& min_arch = {0, 0}) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    std::pair<int, int> arch = GetDeviceGPUArch(device_properties);
    if (device_properties.type() == "GPU" && arch >= min_arch) {
      num_gpus++;
    }
  }
  return num_gpus;
}

}  // end namespace

Status AutoMixedPrecision::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }

  // Start by copying input graph to output.
  *output = item.graph;

  int num_gpus = ShouldIgnorePerformance() ? GetNumGPUs(*cluster)
                                           : GetNumGPUs(*cluster, kMinGPUArch);
  if (num_gpus < 1) {
    // AutoMixedPrecision is currently only tuned for GPU.
    LOG(WARNING) << "No (suitable) GPUs detected, skipping " << name()
                 << " graph optimizer";
    return Status::OK();
  }

  // Optimize the output graph in-place.
  AutoMixedPrecisionImpl optimizer(cluster, item.NodesToPreserve(), output,
                                   item.id);
  if (item.id == "tf_graph") {
    LOG(INFO) << "Running " << name() << " graph optimizer";
  } else {
    VLOG(1) << "Running " << name() << " graph optimizer on " << item.id;
  }
  Status status = optimizer.Optimize();
  if (!status.ok()) {
    // Restore the original graph.
    *output = item.graph;
    LOG(WARNING) << name() << " graph optimizer FAILED: " << status.ToString();
  }
  return status;
}

void AutoMixedPrecision::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // Nothing to do for AutoMixedPrecision.
}

}  // end namespace grappler
}  // end namespace tensorflow
