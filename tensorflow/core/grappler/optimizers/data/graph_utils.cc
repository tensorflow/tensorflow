/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include <cstddef>

#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

constexpr char kConstOpName[] = "Const";
constexpr char kRetValOp[] = "_Retval";

constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
constexpr char kToutputTypes[] = "Toutput_types";

template <typename Predicate, typename Collection>
std::vector<int> GetElementIndicesWithPredicate(const Predicate& predicate,
                                                const Collection& collection) {
  std::vector<int> indices = {};
  unsigned idx = 0;
  for (auto&& element : collection) {
    if (predicate(element)) {
      indices.push_back(idx);
    }
    idx++;
  }
  return indices;
}

std::vector<int> CreateNameIndex(const GraphDef& graph) {
  std::map<string, int> names;
  for (int i = 0; i < graph.node_size(); ++i) {
    names[graph.node(i).name()] = i;
  }
  std::vector<int> index(graph.node_size());
  int i = 0;
  for (const auto& pair : names) {
    index[i++] = pair.second;
  }
  return index;
}

std::vector<int> CreateInputIndex(const NodeDef& node) {
  std::map<string, int> inputs;
  for (int i = 0; i < node.input_size(); ++i) {
    inputs[node.input(i)] = i;
  }
  std::vector<int> index(node.input_size());
  int i = 0;
  for (const auto& pair : inputs) {
    index[i++] = pair.second;
  }
  return index;
}

NodeDef* AddScalarConstNodeHelper(
    DataType dtype, const std::function<void(TensorProto*)>& add_value,
    MutableGraphView* graph) {
  NodeDef node;
  node.set_op(kConstOpName);
  SetUniqueGraphNodeName(kConstOpName, graph->graph(), &node);

  (*node.mutable_attr())["dtype"].set_type(dtype);
  std::unique_ptr<tensorflow::TensorProto> tensor =
      std::make_unique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      std::make_unique<tensorflow::TensorShapeProto>();
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  tensor->set_dtype(dtype);
  add_value(tensor.get());
  (*node.mutable_attr())["value"].set_allocated_tensor(tensor.release());

  return graph->AddNode(std::move(node));
}

}  // namespace

NodeDef* AddScalarPlaceholder(DataType dtype, MutableGraphView* graph) {
  NodeDef node;
  node.set_op("Placeholder");
  SetUniqueGraphNodeName(node.op(), graph->graph(), &node);
  (*node.mutable_attr())["dtype"].set_type(dtype);
  TensorShapeProto* shape = (*node.mutable_attr())["shape"].mutable_shape();
  shape->set_unknown_rank(false);
  return graph->AddNode(std::move(node));
}

NodeDef* AddNode(absl::string_view name, absl::string_view op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 MutableGraphView* graph) {
  NodeDef node;
  if (!name.empty()) {
    node.set_name(string(name));
  } else {
    SetUniqueGraphNodeName(op, graph->graph(), &node);
  }
  node.set_op(string(op));
  for (const string& input : inputs) {
    node.add_input(input);
  }
  for (const auto& attr : attributes) {
    (*node.mutable_attr())[attr.first] = attr.second;
  }
  return graph->AddNode(std::move(node));
}

template <>
NodeDef* AddScalarConstNode(bool v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_BOOL, [v](TensorProto* proto) { proto->add_bool_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(double v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_DOUBLE, [v](TensorProto* proto) { proto->add_double_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(float v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_FLOAT, [v](TensorProto* proto) { proto->add_float_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(int v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_INT32, [v](TensorProto* proto) { proto->add_int_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(int64_t v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_INT64, [v](TensorProto* proto) { proto->add_int64_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(absl::string_view v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_STRING,
      [v](TensorProto* proto) { proto->add_string_val(v.data(), v.size()); },
      graph);
}

absl::Status GetScalarConstNodeValueHelper(
    const NodeDef& node, DataType dtype,
    const std::function<void(const Tensor&)>& get_value) {
  if (node.op() != kConstOpName)
    return errors::InvalidArgument("Node ", node.name(),
                                   " is not a Const node. Op: ", node.op());

  Tensor tensor;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &tensor));
  if (!TensorShapeUtils::IsScalar(tensor.shape())) {
    return errors::InvalidArgument(
        "Node ", node.name(),
        " should be a scalar but has shape: ", tensor.shape());
  }

  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument(
        "Node ", node.name(), " should have type ", DataTypeString(dtype),
        " but has type: ", DataTypeString(tensor.dtype()));
  }

  get_value(tensor);

  return absl::OkStatus();
}

template <>
absl::Status GetScalarConstNodeValue(const NodeDef& node, int64_t* value) {
  return GetScalarConstNodeValueHelper(
      node, DT_INT64,
      [value](const Tensor& tensor) { *value = tensor.scalar<int64_t>()(); });
}

template <>
absl::Status GetScalarConstNodeValue(const NodeDef& node, bool* value) {
  return GetScalarConstNodeValueHelper(
      node, DT_BOOL,
      [value](const Tensor& tensor) { *value = tensor.scalar<bool>()(); });
}

bool Compare(const GraphDef& g1, const GraphDef& g2) {
  if (g1.node_size() != g2.node_size()) {
    return false;
  }
  std::vector<int> name_index1 = CreateNameIndex(g1);
  std::vector<int> name_index2 = CreateNameIndex(g2);
  for (int i = 0; i < g1.node_size(); ++i) {
    int idx1 = name_index1[i];
    int idx2 = name_index2[i];
    if (g1.node(idx1).op() != g2.node(idx2).op()) {
      return false;
    }
    if (g1.node(idx1).name() != g2.node(idx2).name()) {
      return false;
    }
    if (g1.node(idx1).input_size() != g2.node(idx2).input_size()) {
      return false;
    }
    std::vector<int> input_index1 = CreateInputIndex(g1.node(idx1));
    std::vector<int> input_index2 = CreateInputIndex(g2.node(idx2));
    for (int j = 0; j < g1.node(idx1).input_size(); ++j) {
      if (!IsSameInput(g1.node(idx1).input(input_index1[j]),
                       g2.node(idx2).input(input_index2[j]))) {
        return false;
      }
    }
  }
  return true;
}

bool ContainsGraphFunctionWithName(absl::string_view name,
                                   const FunctionDefLibrary& library) {
  return FindGraphFunctionWithName(name, library) != -1;
}

bool ContainsGraphNodeWithName(absl::string_view name, const GraphDef& graph) {
  return FindGraphNodeWithName(name, graph) != -1;
}

bool ContainsNodeWithOp(absl::string_view op, const GraphDef& graph) {
  return FindGraphNodeWithOp(op, graph) != -1;
}

int FindGraphFunctionWithName(absl::string_view name,
                              const FunctionDefLibrary& library) {
  return GetFirstElementIndexWithPredicate(
      [&name](const FunctionDef& function) {
        return function.signature().name() == name;
      },
      library.function());
}

int FindGraphNodeWithName(absl::string_view name, const GraphDef& graph) {
  return GetFirstElementIndexWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      graph.node());
}

int FindGraphNodeWithOp(absl::string_view op, const GraphDef& graph) {
  return GetFirstElementIndexWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; }, graph.node());
}

std::vector<int> FindAllGraphNodesWithOp(const string& op,
                                         const GraphDef& graph) {
  return GetElementIndicesWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; }, graph.node());
}

NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph) {
  if (node.input_size() == 0) return nullptr;
  MutableGraphView::InputPort input_port = graph.GetInputPort(node.name(), 0);
  return graph.GetRegularFanin(input_port).node;
}

NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph,
                      int64_t i) {
  if (node.input_size() <= i) return nullptr;
  MutableGraphView::InputPort input_port = graph.GetInputPort(node.name(), i);
  return graph.GetRegularFanin(input_port).node;
}

absl::Status GetDatasetOutputTypesAttr(const NodeDef& node,
                                       DataTypeVector* output_types) {
  // We don't name the output_types attr consistently, so should check for both.
  for (const string& attr_name : {"output_types", "Toutput_types"}) {
    if (node.attr().contains(attr_name)) {
      return GetNodeAttr(node, attr_name, output_types);
    }
  }
  return errors::InvalidArgument("Could not find output_types attr for node: ",
                                 node.name(), " with op: ", node.op());
}

void SetUniqueGraphNodeName(absl::string_view prefix, GraphDef* graph,
                            NodeDef* node) {
  string name = string(prefix);
  int id = graph->node_size();
  while (ContainsGraphNodeWithName(name, *graph)) {
    if (name.rfind("_generated") != string::npos &&
        (name.rfind("_generated") == (name.size() - strlen("_generated")))) {
      name.insert(name.rfind("_generated"), strings::StrCat("/_", id));
    } else {
      name = strings::StrCat(prefix, "/_", id);
    }
    ++id;
  }
  node->set_name(std::move(name));
}

void SetUniqueGraphFunctionName(absl::string_view prefix,
                                const FunctionDefLibrary* library,
                                FunctionDef* function) {
  string name = string(prefix);
  int id = library->function_size();
  while (ContainsGraphFunctionWithName(name, *library)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  function->mutable_signature()->set_name(std::move(name));
}

void CopyAttribute(const string& attribute_name, const NodeDef& from,
                   NodeDef* to_node) {
  (*to_node->mutable_attr())[attribute_name] = from.attr().at(attribute_name);
}

void ConcatAttributeList(const string& attribute_name, const NodeDef& first,
                         const NodeDef& second, NodeDef* to_node) {
  CopyAttribute(attribute_name, first, to_node);
  (*to_node->mutable_attr())
      .at(attribute_name)
      .mutable_list()
      ->MergeFrom(second.attr().at(attribute_name).list());
}

absl::Status EnsureNodeNamesUnique(Graph* g) {
  // Modeled after Scope::Impl::GetUniqueName
  std::unordered_map<string, int> name_map;

  for (auto node : g->op_nodes()) {
    const string& prefix = node->name();
    if (auto entry = gtl::FindOrNull(name_map, prefix)) {
      string unique_name;
      do {
        unique_name = strings::StrCat(prefix, "_", ++(*entry));
      } while (name_map.find(unique_name) != name_map.end());
      name_map.insert({unique_name, 0});
      node->set_name(std::move(unique_name));
    } else {
      name_map.insert({node->name(), 0});
    }
  }

  return absl::OkStatus();
}

absl::Status GetFetchNode(const MutableGraphView& graph,
                          const GrapplerItem& item, NodeDef** fetch_node) {
  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  *fetch_node = graph.GetNode(item.fetch.at(0));

  return absl::OkStatus();
}

bool IsItemDerivedFromFunctionDef(const GrapplerItem& item,
                                  const MutableGraphView& graph_view) {
  for (const auto& fetch_name : item.fetch) {
    auto fetch = graph_view.GetNode(fetch_name);
    if (fetch != nullptr && fetch->op() != kRetValOp) {
      // We found a fetch node which is not a `Retval` op.
      return false;
    }
  }
  // All fetch nodes are `Retval` ops (or we don't have any fetch nodes).
  return true;
}

void MaybeSetFusedMetadata(const NodeDef& node1, const NodeDef& node2,
                           NodeDef* fused_node) {
  data::Metadata metadata1;
  if (node1.attr().contains("metadata")) {
    metadata1.ParseFromString(node1.attr().at("metadata").s());
  }
  data::Metadata metadata2;
  if (node2.attr().contains("metadata")) {
    metadata2.ParseFromString(node2.attr().at("metadata").s());
  }
  data::Metadata fused_metadata;
  auto normalize_name = [](const string& name) {
    return name.empty() ? "?" : name;
  };
  *fused_metadata.mutable_name() =
      strings::StrCat("fused(", normalize_name(metadata1.name()), ",",
                      normalize_name(metadata2.name()), ")");
  fused_metadata.SerializeToString(
      (*fused_node->mutable_attr())["metadata"].mutable_s());
}

bool CopyShapesAndTypesAttrs(const NodeDef& from, NodeDef* to_node) {
  auto* attr = gtl::FindOrNull(from.attr(), kOutputTypes);
  attr = (attr == nullptr ? gtl::FindOrNull(from.attr(), kToutputTypes) : attr);

  if (attr == nullptr) return false;
  (*to_node->mutable_attr())[kOutputTypes] = *attr;

  attr = gtl::FindOrNull(from.attr(), kOutputShapes);
  if (attr == nullptr) return false;
  (*to_node->mutable_attr())[kOutputShapes] = *attr;
  return true;
}

namespace {
const auto* kSloppyAttrOps = new absl::flat_hash_set<string>{
    "ParallelInterleaveDatasetV2",
    "ParallelMapDataset",
    "ParseExampleDataset",
};

const auto* kReplicateOnSplitAttrOps = new absl::flat_hash_set<string>{
    "TensorSliceDataset",
    "RangeDataset",
};

const auto* kDeterministicAttrOps = new absl::flat_hash_set<string>{
    "LegacyParallelInterleaveDatasetV2",
    "ParallelInterleaveDatasetV3",
    "ParallelInterleaveDatasetV4",
    "ParallelMapDatasetV2",
    "ParallelBatchDataset",
};
}  // anonymous namespace

bool HasSloppyAttr(const string& op) { return kSloppyAttrOps->contains(op); }

bool HasReplicateOnSplitAttr(const string& op) {
  return kReplicateOnSplitAttrOps->contains(op);
}

bool HasDeterministicAttr(const string& op) {
  return kDeterministicAttrOps->contains(op);
}

absl::Status SetMetadataName(const std::string& name, NodeDef* node) {
  data::Metadata metadata;
  if (node->attr().contains("metadata")) {
    metadata.ParseFromString(node->attr().at("metadata").s());
  }
  if (!metadata.name().empty()) {
    return errors::InvalidArgument("Node ", node->name(),
                                   " already has a metadata name \"",
                                   metadata.name(), "\".");
  }
  *metadata.mutable_name() = name;
  metadata.SerializeToString((*node->mutable_attr())["metadata"].mutable_s());
  return absl::OkStatus();
}

}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
