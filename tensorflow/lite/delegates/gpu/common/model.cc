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

#include "tensorflow/lite/delegates/gpu/common/model.h"

#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

std::vector<Node*> GraphFloat32::nodes() const {
  return FilterNodes([](const NodeDef&) { return true; });
}

std::vector<Value*> GraphFloat32::values() const {
  return FilterValues([](const ValueDef&) { return true; });
}

std::vector<Value*> GraphFloat32::inputs() const {
  return FilterValues([](const ValueDef& v) { return v.producer == nullptr; });
}

std::vector<Value*> GraphFloat32::variable_inputs() const {
  return FilterValues(
      [](const ValueDef& v) { return v.value->tensor.is_variable_input; });
}

std::vector<Value*> GraphFloat32::outputs() const {
  return FilterValues([](const ValueDef& v) { return v.consumers.empty(); });
}

std::vector<Value*> GraphFloat32::FindInputs(NodeId id) const {
  if (id >= nodes_.size()) {
    return {};
  }
  return nodes_.at(id).inputs;
}

std::vector<Value*> GraphFloat32::FindOutputs(NodeId id) const {
  if (id >= nodes_.size()) {
    return {};
  }
  return nodes_.at(id).outputs;
}

bool GraphFloat32::IsGraphInput(ValueId id) const {
  if (id >= values_.size()) {
    return false;
  }
  return values_[id].producer == nullptr;
}

bool GraphFloat32::IsGraphOutput(ValueId id) const {
  if (id >= values_.size()) {
    return false;
  }
  return values_[id].consumers.empty();
}

Node* GraphFloat32::FindProducer(ValueId id) const {
  if (id >= values_.size()) {
    return nullptr;
  }
  return values_[id].producer;
}

std::vector<Node*> GraphFloat32::FindConsumers(ValueId id) const {
  if (id >= values_.size()) {
    return {};
  }
  return values_[id].consumers;
}

Node* GraphFloat32::GetNode(NodeId id) const {
  if (id >= nodes_.size()) {
    return {};
  }
  return nodes_.at(id).node.get();
}

Value* GraphFloat32::GetValue(ValueId id) const {
  if (id >= values_.size()) {
    return nullptr;
  }
  return values_[id].value.get();
}

Node* GraphFloat32::NewNode() {
  const NodeId new_id = nodes_.size();
  NodeDef def;
  def.node = absl::make_unique<Node>(Node{static_cast<NodeId>(new_id), {}});
  Node* node = def.node.get();
  nodes_[new_id] = std::move(def);
  execution_plan_.push_back(new_id);
  return node;
}

absl::Status GraphFloat32::InsertNodeAfter(NodeId id, Node** new_node) {
  if (id >= nodes_.size()) {
    return absl::OutOfRangeError("NodeId is out of range");
  }
  int idx = 0;
  while (idx < execution_plan_.size()) {
    if (execution_plan_[idx] == id) break;
    ++idx;
  }
  if (idx == execution_plan_.size()) {
    return absl::OutOfRangeError("NodeId not in execution plan");
  }

  const NodeId new_id = nodes_.size();
  NodeDef def;
  def.node = absl::make_unique<Node>(Node{static_cast<NodeId>(new_id), {}});
  *new_node = def.node.get();
  nodes_[new_id] = std::move(def);
  execution_plan_.insert(execution_plan_.begin() + idx + 1, new_id);
  return absl::OkStatus();
}

Value* GraphFloat32::NewValue() {
  ValueDef def;
  def.value =
      absl::make_unique<Value>(Value{static_cast<ValueId>(values_.size()), {}});
  Value* value = def.value.get();
  values_.push_back(std::move(def));
  return value;
}

absl::Status GraphFloat32::SetProducer(NodeId producer, ValueId value) {
  ValueDef* v;
  RETURN_IF_ERROR(LookupValue(value, &v));
  Value* value_ptr = v->value.get();
  NodeDef* n;
  RETURN_IF_ERROR(LookupNode(producer, &n));
  Node* node_ptr = n->node.get();

  // check if this value has the same producer already
  if (node_ptr == v->producer) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Node ", producer, " is already a producer of the value ", value));
  }

  // Check if the node is a consumer of this value.
  if (IsInput(producer, value)) {
    return absl::InvalidArgumentError("Node is a consumer of the value");
  }

  if (v->producer != nullptr) {
    // value is no longer produced by it's previous producer.
    Erase(&nodes_[v->producer->id].outputs, value_ptr);
  }
  v->producer = node_ptr;
  n->outputs.push_back(value_ptr);
  return absl::OkStatus();
}

absl::Status GraphFloat32::RemoveProducer(ValueId value) {
  ValueDef* v;
  RETURN_IF_ERROR(LookupValue(value, &v));
  Value* value_ptr = v->value.get();
  if (v->producer == nullptr) {
    return absl::InvalidArgumentError("Value does not have a producer");
  }
  Erase(&nodes_[v->producer->id].outputs, value_ptr);
  v->producer = nullptr;
  return absl::OkStatus();
}

absl::Status GraphFloat32::AddConsumer(NodeId consumer, ValueId value) {
  ValueDef* v;
  RETURN_IF_ERROR(LookupValue(value, &v));
  Value* value_ptr = v->value.get();
  NodeDef* n;
  RETURN_IF_ERROR(LookupNode(consumer, &n));
  Node* node_ptr = n->node.get();

  // check if this value has the same producer already
  if (node_ptr == v->producer) {
    return absl::InvalidArgumentError("Node is a producer of the value");
  }

  // check if this value has the same consumer already
  if (IsInput(consumer, value)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Node ", consumer, " is already a consumer of the value ", value));
  }

  n->inputs.push_back(value_ptr);
  v->consumers.push_back(node_ptr);
  return absl::OkStatus();
}

// Replace input value for given node.
absl::Status GraphFloat32::ReplaceInput(NodeId node, ValueId old_value,
                                        ValueId new_value) {
  ValueDef* v_old;
  RETURN_IF_ERROR(LookupValue(old_value, &v_old));
  Value* value_old_ptr = v_old->value.get();
  ValueDef* v_new;
  RETURN_IF_ERROR(LookupValue(new_value, &v_new));
  Value* value_new_ptr = v_new->value.get();
  NodeDef* n;
  RETURN_IF_ERROR(LookupNode(node, &n));
  Node* node_ptr = n->node.get();

  // Check if the node is a consumer of old_value.
  if (!IsInput(node, old_value)) {
    return absl::InvalidArgumentError("old_value must be input of node.");
  }

  // Check if the node is not a consumer of new_value.
  if (IsInput(node, new_value)) {
    return absl::InvalidArgumentError("new_value can not be input of node.");
  }

  // Check if this value has the same producer already
  if (node_ptr == v_new->producer) {
    return absl::InvalidArgumentError("new_value can not be output of node.");
  }

  for (int i = 0; i < n->inputs.size(); ++i) {
    if (n->inputs[i] == value_old_ptr) {
      n->inputs[i] = value_new_ptr;
      break;
    }
  }
  v_new->consumers.push_back(node_ptr);
  Erase(&v_old->consumers, node_ptr);
  return absl::OkStatus();
}

absl::Status GraphFloat32::RemoveConsumer(NodeId consumer, ValueId value) {
  ValueDef* v;
  RETURN_IF_ERROR(LookupValue(value, &v));
  Value* value_ptr = v->value.get();
  NodeDef* n;
  RETURN_IF_ERROR(LookupNode(consumer, &n));
  Node* node_ptr = n->node.get();
  if (!IsInput(consumer, value)) {
    return absl::InvalidArgumentError("Node is not a consumer of the value");
  }
  Erase(&n->inputs, value_ptr);
  Erase(&v->consumers, node_ptr);
  return absl::OkStatus();
}

absl::Status GraphFloat32::DeleteNode(NodeId id) {
  NodeDef* n;
  RETURN_IF_ERROR(LookupNode(id, &n));
  Node* node_ptr = n->node.get();
  for (auto value : n->inputs) {
    Erase(&values_[value->id].consumers, node_ptr);
  }
  for (auto value : n->outputs) {
    values_[value->id].producer = nullptr;
  }
  n->inputs.clear();
  n->outputs.clear();
  n->node.reset();
  return absl::OkStatus();
}

absl::Status GraphFloat32::DeleteValue(ValueId id) {
  ValueDef* v;
  RETURN_IF_ERROR(LookupValue(id, &v));
  Value* value_ptr = v->value.get();
  if (v->producer != nullptr) {
    Erase(&nodes_[v->producer->id].outputs, value_ptr);
  }
  if (!v->consumers.empty()) {
    for (auto node : v->consumers) {
      Erase(&nodes_[node->id].inputs, value_ptr);
    }
  }
  v->producer = nullptr;
  v->consumers.clear();
  v->value.reset();
  return absl::OkStatus();
}

absl::Status GraphFloat32::MakeExactCopy(GraphFloat32* model) const {
  model->nodes_.clear();
  model->execution_plan_.clear();
  model->values_.clear();
  for (auto& value_def : values_) {
    model->values_.push_back({});
    if (value_def.value) {
      model->values_.back().value = absl::make_unique<Value>(*value_def.value);
    }
  }
  // Add all nodes first.
  for (auto node_id : execution_plan_) {
    model->execution_plan_.push_back(node_id);
    model->nodes_[node_id] = {};
    auto& node_def = nodes_.at(node_id);
    if (node_def.node) {
      model->nodes_[node_id].node = absl::make_unique<Node>(*node_def.node);
    }
  }
  // Wire up dependencies between nodes.
  for (auto node_id : execution_plan_) {
    auto& node_def = nodes_.at(node_id);
    if (node_def.node) {
      for (auto output : node_def.outputs) {
        RETURN_IF_ERROR(model->SetProducer(node_def.node->id, output->id));
      }
      for (auto input : node_def.inputs) {
        RETURN_IF_ERROR(model->AddConsumer(node_def.node->id, input->id));
      }
    }
  }
  return absl::OkStatus();
}

bool GraphFloat32::IsInput(NodeId node, ValueId value) {
  if (node >= nodes_.size() || value >= values_.size()) {
    return false;
  }
  const NodeDef& n = nodes_[node];
  const ValueDef& v = values_[value];
  if (!n.node || !v.value) {
    return false;
  }
  return std::find(n.inputs.begin(), n.inputs.end(), v.value.get()) !=
         n.inputs.end();
}

absl::Status GraphFloat32::LookupNode(NodeId id, NodeDef** node_def) {
  if (id >= nodes_.size()) {
    return absl::OutOfRangeError("NodeId is out of range");
  }
  auto& n = nodes_[id];
  if (!n.node) {
    return absl::OutOfRangeError("Node is already deleted");
  }
  *node_def = &n;
  return absl::OkStatus();
}

absl::Status GraphFloat32::LookupValue(ValueId id, ValueDef** value_def) {
  if (id >= values_.size()) {
    return absl::OutOfRangeError("ValueId is out of range");
  }
  auto& v = values_[id];
  if (!v.value) {
    return absl::OutOfRangeError("Value is already deleted");
  }
  *value_def = &v;
  return absl::OkStatus();
}

absl::Status RemovePrecedingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep) {
  // Make sure all outputs from to_remove are consumed by to_keep.
  for (auto output : graph->FindOutputs(to_remove->id)) {
    auto consumers = graph->FindConsumers(output->id);
    if (consumers.size() > 1 ||
        (consumers.size() == 1 && consumers[0] != to_keep)) {
      return absl::InvalidArgumentError(
          "Output from to_remove node has other consumers");
    }
  }

  // Update all references
  for (auto input : graph->FindInputs(to_remove->id)) {
    RETURN_IF_ERROR(graph->AddConsumer(to_keep->id, input->id));
  }
  for (auto output : graph->FindOutputs(to_remove->id)) {
    RETURN_IF_ERROR(graph->DeleteValue(output->id));
  }
  return graph->DeleteNode(to_remove->id);
}

absl::Status RemoveFollowingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep) {
  // Make sure all inputs to to_remove are produced by to_keep.
  for (auto input : graph->FindInputs(to_remove->id)) {
    Node* producer = graph->FindProducer(input->id);
    if (producer->id != to_keep->id) {
      return absl::InvalidArgumentError("To_remove node has other inputs");
    }
  }

  for (auto input : graph->FindInputs(to_remove->id)) {
    RETURN_IF_ERROR(graph->DeleteValue(input->id));
  }
  for (auto output : graph->FindOutputs(to_remove->id)) {
    RETURN_IF_ERROR(graph->SetProducer(to_keep->id, output->id));
  }
  return graph->DeleteNode(to_remove->id);
}

absl::Status RemoveSimpleNodeKeepInput(GraphFloat32* graph,
                                       const Node* simple_node) {
  const auto inputs = graph->FindInputs(simple_node->id);
  const auto outputs = graph->FindOutputs(simple_node->id);
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::FailedPreconditionError(
        "simple_node node must have 1 input and 1 output");
  }
  const auto input_id = inputs[0]->id;
  const auto output_id = outputs[0]->id;
  const Node* producer = graph->FindProducer(input_id);
  const auto consumers = graph->FindConsumers(output_id);
  RETURN_IF_ERROR(graph->DeleteNode(simple_node->id));
  for (auto& consumer : consumers) {
    RETURN_IF_ERROR(graph->ReplaceInput(consumer->id, output_id, input_id));
  }
  RETURN_IF_ERROR(graph->DeleteValue(output_id));
  if (!producer && consumers.empty()) {
    RETURN_IF_ERROR(graph->DeleteValue(input_id));
  }
  return absl::OkStatus();
}

absl::Status RemoveSimpleNodeKeepOutput(GraphFloat32* graph,
                                        const Node* simple_node) {
  const auto inputs = graph->FindInputs(simple_node->id);
  const auto outputs = graph->FindOutputs(simple_node->id);
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::FailedPreconditionError(
        "simple_node must have 1 input and 1 output");
  }
  const auto input_id = inputs[0]->id;
  const auto output_id = outputs[0]->id;
  const Node* producer = graph->FindProducer(input_id);
  const auto input_consumers = graph->FindConsumers(input_id);
  if (input_consumers.size() != 1) {
    return absl::FailedPreconditionError(
        "simple_node should be the only consumer on the node.");
  }

  RETURN_IF_ERROR(graph->DeleteNode(simple_node->id));
  if (producer) {
    RETURN_IF_ERROR(graph->RemoveProducer(input_id));
    RETURN_IF_ERROR(graph->SetProducer(producer->id, output_id));
  }

  RETURN_IF_ERROR(graph->DeleteValue(input_id));

  const auto output_consumers = graph->FindConsumers(output_id);
  if (!producer && output_consumers.empty()) {
    RETURN_IF_ERROR(graph->DeleteValue(output_id));
  }
  return absl::OkStatus();
}

absl::Status AddOutput(GraphFloat32* graph, const Node* from_node,
                       Value** output) {
  auto link = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(from_node->id, link->id));
  *output = link;
  return absl::OkStatus();
}

absl::Status ConnectTwoNodes(GraphFloat32* graph, const Node* from_node,
                             const Node* to_node, Value** output) {
  const Node* output_producer =
      *output ? graph->FindProducer((*output)->id) : nullptr;
  // Output is already initialized, but producer is not from_node.
  if (*output && output_producer && output_producer->id != from_node->id) {
    return absl::InvalidArgumentError("Wrong output is passed.");
  }
  // Output is already initialized, and producer is from_node.
  if (*output) {
    RETURN_IF_ERROR(graph->AddConsumer(to_node->id, (*output)->id));
  } else {
    // Output is not initialized.
    Value* link;
    RETURN_IF_ERROR(AddOutput(graph, from_node, &link));
    RETURN_IF_ERROR(graph->AddConsumer(to_node->id, link->id));
    *output = link;
  }
  return absl::OkStatus();
}

absl::Status CheckBatchSizeForAllValues(const GraphFloat32& model) {
  if (model.values().empty()) return absl::OkStatus();
  const int32_t b = model.values()[0]->tensor.shape.b;
  for (auto value : model.values()) {
    if (value->tensor.shape.b != b) {
      return absl::InvalidArgumentError(
          absl::StrCat("Batch size mismatch, expected ", b, " but got ",
                       value->tensor.shape.b));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
