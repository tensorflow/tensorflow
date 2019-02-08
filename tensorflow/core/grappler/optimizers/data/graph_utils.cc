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

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

constexpr char kConstOpName[] = "Const";

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
      tensorflow::MakeUnique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      tensorflow::MakeUnique<tensorflow::TensorShapeProto>();
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

NodeDef* AddNode(StringPiece name, StringPiece op,
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
  for (auto attr : attributes) {
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
NodeDef* AddScalarConstNode(int64 v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_INT64, [v](TensorProto* proto) { proto->add_int64_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(StringPiece v, MutableGraphView* graph) {
  return AddScalarConstNodeHelper(
      DT_STRING,
      [v](TensorProto* proto) { proto->add_string_val(v.data(), v.size()); },
      graph);
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

bool ContainsGraphFunctionWithName(StringPiece name,
                                   const FunctionDefLibrary& library) {
  return FindGraphFunctionWithName(name, library) != -1;
}

bool ContainsGraphNodeWithName(StringPiece name, const GraphDef& graph) {
  return FindGraphNodeWithName(name, graph) != -1;
}

bool ContainsNodeWithOp(StringPiece op, const GraphDef& graph) {
  return FindGraphNodeWithOp(op, graph) != -1;
}

int FindGraphFunctionWithName(StringPiece name,
                              const FunctionDefLibrary& library) {
  return GetFirstElementIndexWithPredicate(
      [&name](const FunctionDef& function) {
        return function.signature().name() == name;
      },
      library.function());
}

int FindGraphNodeWithName(StringPiece name, const GraphDef& graph) {
  return GetFirstElementIndexWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      graph.node());
}

int FindGraphNodeWithOp(StringPiece op, const GraphDef& graph) {
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
                      int64 i) {
  if (node.input_size() <= i) return nullptr;
  MutableGraphView::InputPort input_port = graph.GetInputPort(node.name(), i);
  return graph.GetRegularFanin(input_port).node;
}

void SetUniqueGraphNodeName(StringPiece prefix, GraphDef* graph,
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

void SetUniqueGraphFunctionName(StringPiece prefix, FunctionDefLibrary* library,
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

Status EnsureNodeNamesUnique(Graph* g) {
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

  return Status::OK();
}
}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
