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
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

constexpr char kConstOpName[] = "Const";

int FindNodeWithPredicate(const std::function<bool(const NodeDef&)>& predicate,
                          const GraphDef& graph) {
  for (int i = 0; i < graph.node_size(); ++i) {
    if (predicate(graph.node(i))) {
      return i;
    }
  }
  return -1;
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

Status AddScalarConstNodeHelper(
    DataType dtype, const std::function<void(TensorProto*)>& add_value,
    GraphDef* graph, NodeDef** result) {
  NodeDef* node = graph->add_node();
  node->set_op(kConstOpName);
  SetUniqueName(kConstOpName, graph, node);
  (*node->mutable_attr())["dtype"].set_type(dtype);
  std::unique_ptr<tensorflow::TensorProto> tensor =
      tensorflow::MakeUnique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      tensorflow::MakeUnique<tensorflow::TensorShapeProto>();
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  tensor->set_dtype(dtype);
  add_value(tensor.get());
  (*node->mutable_attr())["value"].set_allocated_tensor(tensor.release());
  *result = node;
  return Status::OK();
}

}  // namespace

Status AddNode(const string& name, const string& op,
               const std::vector<string>& inputs,
               const std::vector<std::pair<string, AttrValue>>& attributes,
               GraphDef* graph, NodeDef** result) {
  NodeDef* node = graph->add_node();
  if (!name.empty()) {
    node->set_name(name);
  } else {
    SetUniqueName(op, graph, node);
  }
  node->set_op(op);
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (auto attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  *result = node;
  return Status::OK();
}

template <>
Status AddScalarConstNode(bool v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_BOOL, [v](TensorProto* proto) { proto->add_bool_val(v); }, graph,
      result);
}

template <>
Status AddScalarConstNode(double v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_DOUBLE, [v](TensorProto* proto) { proto->add_double_val(v); }, graph,
      result);
}

template <>
Status AddScalarConstNode(float v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_FLOAT, [v](TensorProto* proto) { proto->add_float_val(v); }, graph,
      result);
}

template <>
Status AddScalarConstNode(int v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_INT32, [v](TensorProto* proto) { proto->add_int_val(v); }, graph,
      result);
}

template <>
Status AddScalarConstNode(int64 v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_INT64, [v](TensorProto* proto) { proto->add_int64_val(v); }, graph,
      result);
}

template <>
Status AddScalarConstNode(StringPiece v, GraphDef* graph, NodeDef** result) {
  return AddScalarConstNodeHelper(
      DT_STRING,
      [v](TensorProto* proto) { proto->add_string_val(v.data(), v.size()); },
      graph, result);
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

bool ContainsNodeWithName(const string& name, const GraphDef& graph) {
  return FindNodeWithName(name, graph) != -1;
}

bool ContainsNodeWithOp(const string& op, const GraphDef& graph) {
  return FindNodeWithOp(op, graph) != -1;
}

Status DeleteNodes(const std::set<string>& nodes_to_delete, GraphDef* graph) {
  int last = graph->node_size() - 1;
  for (int i = graph->node_size() - 1; i >= 0; --i) {
    const NodeDef& node = graph->node(i);
    if (nodes_to_delete.find(node.name()) != nodes_to_delete.end()) {
      graph->mutable_node()->SwapElements(i, last);
      last--;
    }
  }
  graph->mutable_node()->DeleteSubrange(last + 1,
                                        graph->node_size() - last - 1);
  return Status::OK();
}

int FindNodeWithName(const string& name, const GraphDef& graph) {
  return FindNodeWithPredicate(
      [name](const NodeDef& node) { return node.name() == name; }, graph);
}

int FindNodeWithOp(const string& op, const GraphDef& graph) {
  return FindNodeWithPredicate(
      [op](const NodeDef& node) { return node.op() == op; }, graph);
}

void SetUniqueName(const string& op, GraphDef* graph, NodeDef* node) {
  int id = graph->node_size();
  while (ContainsNodeWithName(strings::StrCat(op, "/_", id), *graph)) {
    ++id;
  }
  node->set_name(strings::StrCat(op, "/_", id));
}

void ReplaceInput(const NodeDef& old_input, const NodeDef& new_input,
                  GraphView* graph) {
  GraphView::OutputPort output_port = graph->GetOutputPort(old_input.name(), 0);
  auto fanout = graph->GetFanout(output_port);
  for (auto& input_port : fanout)
    input_port.node->set_input(0, new_input.name());
}

}  // end namespace graph_utils
}  // end namespace grappler
}  // end namespace tensorflow
