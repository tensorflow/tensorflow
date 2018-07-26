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
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

constexpr char kConstOpName[] = "Const";

template <typename Predicate, typename Collection>
int GetElementIdxWithPredicate(const Predicate& predicate,
                               const Collection& collection) {
  auto it = std::find_if(collection.begin(), collection.end(), predicate);
  if (it == collection.end()) return -1;
  return std::distance(collection.begin(), it);
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
  SetUniqueGraphNodeName(kConstOpName, graph->GetGraph(), &node);

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

NodeDef* AddNode(const string& name, const string& op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 MutableGraphView* graph) {
  NodeDef node;
  if (!name.empty()) {
    node.set_name(name);
  } else {
    SetUniqueGraphNodeName(op, graph->GetGraph(), &node);
  }
  node.set_op(op);
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

bool ContainsGraphNodeWithName(const string& name, const GraphDef& graph) {
  return FindGraphNodeWithName(name, graph) != -1;
}

bool ContainsNodeWithOp(const string& op, const GraphDef& graph) {
  return FindNodeWithOp(op, graph) != -1;
}

bool ContainsGraphFunctionWithName(const string& name,
                                   const FunctionDefLibrary& library) {
  return FindGraphFunctionWithName(name, library) != -1;
}

bool ContainsFunctionNodeWithName(const string& name,
                                  const FunctionDef& function) {
  return FindFunctionNodeWithName(name, function) != -1;
}

int FindGraphNodeWithName(const string& name, const GraphDef& graph) {
  return GetElementIdxWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      graph.node());
}

int FindNodeWithOp(const string& op, const GraphDef& graph) {
  return GetElementIdxWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; }, graph.node());
}

int FindGraphFunctionWithName(const string& name,
                              const FunctionDefLibrary& library) {
  return GetElementIdxWithPredicate(
      [&name](const FunctionDef& function) {
        return function.signature().name() == name;
      },
      library.function());
}

int FindFunctionNodeWithName(const string& name, const FunctionDef& function) {
  return GetElementIdxWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      function.node_def());
}

void SetUniqueGraphNodeName(const string& prefix, GraphDef* graph,
                            NodeDef* node) {
  string name = prefix;
  int id = graph->node_size();
  while (ContainsGraphNodeWithName(name, *graph)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  node->set_name(std::move(name));
}

void SetUniqueFunctionNodeName(const string& prefix, FunctionDef* function,
                               NodeDef* node) {
  string name = prefix;
  int id = function->node_def_size();
  while (ContainsFunctionNodeWithName(name, *function)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  node->set_name(std::move(name));
}

void SetUniqueGraphFunctionName(const string& prefix,
                                FunctionDefLibrary* library,
                                FunctionDef* function) {
  string name = prefix;
  int id = library->function_size();
  while (ContainsGraphFunctionWithName(name, *library)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  function->mutable_signature()->set_name(name);
}

}  // end namespace graph_utils
}  // end namespace grappler
}  // end namespace tensorflow
