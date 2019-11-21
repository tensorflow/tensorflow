/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

// There is yet another representation of CNN graph. The primary purpose of this
// representation is to simplify graph manipulation.

using ValueId = uint32_t;

using NodeId = uint32_t;

// Connects tensor's producer and operation that depends on this tensor.
template <typename TensorT>
struct Value {
  using TensorType = TensorT;

  const ValueId id;

  TensorType tensor;
};

struct Operation {
  std::string type;

  absl::any attributes;
};

struct Node {
  const NodeId id;

  Operation operation;
};

// Graph is DAG that consists of nodes and values. Each value may have a single
// producer node and multiple consumer nodes. Therefore, each node may have
// multiple input and output values.
//
// Value that does not have a producer is a graph's input. Value that does not
// have a consumer is a graph's output.
//
// Interface provides methods for graph introspection and manipulation. Abstract
// interface makes allows subgraphs representation to ensure safe manipulations.
template <typename TensorT>
class Graph {
 public:
  virtual ~Graph() = default;

  // @return a collection of nodes in this graph.
  virtual std::vector<Node*> nodes() const = 0;

  // @return a collection of values in this graph.
  virtual std::vector<Value<TensorT>*> values() const = 0;

  // @return graph inputs, that are values without producers.
  virtual std::vector<Value<TensorT>*> inputs() const = 0;

  // @return graph outputs, that are values without consumers.
  virtual std::vector<Value<TensorT>*> outputs() const = 0;

  // @return inputs into the given node. Returns empty vector for deleted node.
  virtual std::vector<Value<TensorT>*> FindInputs(NodeId id) const = 0;

  // @return outputs from the given node. Returns empty vector for deleted node.
  virtual std::vector<Value<TensorT>*> FindOutputs(NodeId id) const = 0;

  virtual bool IsGraphInput(ValueId id) const = 0;

  virtual bool IsGraphOutput(ValueId id) const = 0;

  // @return producer of the given value. Returns nullptr for deleted value.
  virtual Node* FindProducer(ValueId id) const = 0;

  // @return consumers of the given value. Returns empty vector for deleted
  // value.
  virtual std::vector<Node*> FindConsumers(ValueId id) const = 0;

  // @return a node or nullptr if node with the given id is not present.
  virtual Node* GetNode(NodeId id) const = 0;

  // @return a value or nullptr if value with the given id is not present.
  virtual Value<TensorT>* GetValue(ValueId id) const = 0;

  //////////////////////////////////////////////////////////////////////////////
  // Graph manipulation functions are below
  //////////////////////////////////////////////////////////////////////////////

  // @return new node created in this graph
  // NOTE: nodes should be created in the topological order, e.g. node A that
  // depends on a value from node B should be created after node B.
  virtual Node* NewNode() = 0;

  // @return new value created in this graph
  virtual Value<TensorT>* NewValue() = 0;

  // Sets a producer for the given value. There could be a single producer
  // for a value. If a value had another producer, it will reassign producer
  // appropriately. If a value didn't have a producer, it will be removed
  // from a graph's input.
  virtual Status SetProducer(NodeId producer, ValueId value) = 0;

  // Removes a producer for the given value. Value becomes producer-less and
  // therefore becomes graph's input.
  virtual Status RemoveProducer(ValueId value) = 0;

  // Sets a consumer for the given value. There could be multiple consumers
  // for a value.
  virtual Status AddConsumer(NodeId consumer, ValueId value) = 0;

  // Replace input value for given node.
  virtual Status ReplaceInput(NodeId node, ValueId old_value,
                              ValueId new_value) = 0;

  // Removes a consumer for the given value. If value does not have any
  // consumers it becomes graph's output.
  virtual Status RemoveConsumer(NodeId consumer, ValueId value) = 0;

  // Removes node from this graph. For all input values this node will be
  // removed from consumers and for all output values a producer will be
  // removed.
  virtual Status DeleteNode(NodeId id) = 0;

  // Removes value from this graph. It will be removed from inputs for all
  // dependent nodes. A node that was a producer of this value will loose its
  // output.
  virtual Status DeleteValue(ValueId id) = 0;
};

// Implementation of a Graph interface. It keeps values and nodes referenced by
// their index in a vector. Therefore, nodes and values are never deleted, but
// rather erased, where corresponding index remains.
//
// It is possible to re-use removed indices, but it is not implemented yet.
template <typename TensorT>
class Model : public Graph<TensorT> {
 public:
  const std::string& name() const { return name_; }

  void set_name(std::string name) { name_ = std::move(name); }

  std::vector<Value<TensorT>*> values() const final {
    return FilterValues([](const ValueDef&) { return true; });
  }

  std::vector<Node*> nodes() const final {
    return FilterNodes([](const NodeDef&) { return true; });
  }

  std::vector<Value<TensorT>*> inputs() const final {
    return FilterValues(
        [](const ValueDef& v) { return v.producer == nullptr; });
  }

  std::vector<Value<TensorT>*> outputs() const final {
    return FilterValues([](const ValueDef& v) { return v.consumers.empty(); });
  }

  bool IsGraphInput(ValueId id) const final {
    if (id >= values_.size()) {
      return false;
    }
    return values_[id].producer == nullptr;
  }

  bool IsGraphOutput(ValueId id) const final {
    if (id >= values_.size()) {
      return false;
    }
    return values_[id].consumers.empty();
  }

  Node* GetNode(NodeId id) const final {
    if (id >= nodes_.size()) {
      return {};
    }
    return nodes_[id].node.get();
  }

  Value<TensorT>* GetValue(ValueId id) const final {
    if (id >= values_.size()) {
      return nullptr;
    }
    return values_[id].value.get();
  }

  Node* NewNode() final {
    NodeDef def;
    def.node =
        absl::make_unique<Node>(Node{static_cast<NodeId>(nodes_.size()), {}});
    Node* node = def.node.get();
    nodes_.push_back(std::move(def));
    return node;
  }

  Value<TensorT>* NewValue() final {
    ValueDef def;
    def.value = absl::make_unique<Value<TensorT>>(
        Value<TensorT>{static_cast<ValueId>(values_.size()), {}});
    Value<TensorT>* value = def.value.get();
    values_.push_back(std::move(def));
    return value;
  }

  std::vector<Value<TensorT>*> FindInputs(NodeId id) const final {
    if (id >= nodes_.size()) {
      return {};
    }
    return nodes_[id].inputs;
  }

  std::vector<Value<TensorT>*> FindOutputs(NodeId id) const final {
    if (id >= nodes_.size()) {
      return {};
    }
    return nodes_[id].outputs;
  }

  Node* FindProducer(ValueId id) const final {
    if (id >= values_.size()) {
      return nullptr;
    }
    return values_[id].producer;
  }

  std::vector<Node*> FindConsumers(ValueId id) const final {
    if (id >= values_.size()) {
      return {};
    }
    return values_[id].consumers;
  }

  Status SetProducer(NodeId producer, ValueId value) final {
    ValueDef* v;
    RETURN_IF_ERROR(LookupValue(value, &v));
    Value<TensorT>* value_ptr = v->value.get();
    NodeDef* n;
    RETURN_IF_ERROR(LookupNode(producer, &n));
    Node* node_ptr = n->node.get();

    // check if this value has the same producer already
    if (node_ptr == v->producer) {
      return InvalidArgumentError("Node is already a producer of the value");
    }

    // Check if the node is a consumer of this value.
    if (IsInput(producer, value)) {
      return InvalidArgumentError("Node is a consumer of the value");
    }
    // TODO(akulik): detect circular dependency?

    if (v->producer != nullptr) {
      // value is no longer produced by it's previous producer.
      Erase(&nodes_[v->producer->id].outputs, value_ptr);
    }
    v->producer = node_ptr;
    n->outputs.push_back(value_ptr);
    return OkStatus();
  }

  Status RemoveProducer(ValueId value) final {
    ValueDef* v;
    RETURN_IF_ERROR(LookupValue(value, &v));
    Value<TensorT>* value_ptr = v->value.get();
    if (v->producer == nullptr) {
      return InvalidArgumentError("Value does not have a producer");
    }
    Erase(&nodes_[v->producer->id].outputs, value_ptr);
    v->producer = nullptr;
    return OkStatus();
  }

  Status ReplaceInput(NodeId node, ValueId old_value, ValueId new_value) final {
    ValueDef* v_old;
    RETURN_IF_ERROR(LookupValue(old_value, &v_old));
    Value<TensorT>* value_old_ptr = v_old->value.get();
    ValueDef* v_new;
    RETURN_IF_ERROR(LookupValue(new_value, &v_new));
    Value<TensorT>* value_new_ptr = v_new->value.get();
    NodeDef* n;
    RETURN_IF_ERROR(LookupNode(node, &n));
    Node* node_ptr = n->node.get();

    // Check if the node is a consumer of old_value.
    if (!IsInput(node, old_value)) {
      return InvalidArgumentError("old_value must be input of node.");
    }

    // Check if the node is not a consumer of new_value.
    if (IsInput(node, new_value)) {
      return InvalidArgumentError("new_value can not be input of node.");
    }

    // Check if this value has the same producer already
    if (node_ptr == v_new->producer) {
      return InvalidArgumentError("new_value can not be output of node.");
    }

    for (int i = 0; i < n->inputs.size(); ++i) {
      if (n->inputs[i] == value_old_ptr) {
        n->inputs[i] = value_new_ptr;
        break;
      }
    }
    v_new->consumers.push_back(node_ptr);
    Erase(&v_old->consumers, node_ptr);
    return OkStatus();
  }

  Status AddConsumer(NodeId consumer, ValueId value) final {
    ValueDef* v;
    RETURN_IF_ERROR(LookupValue(value, &v));
    Value<TensorT>* value_ptr = v->value.get();
    NodeDef* n;
    RETURN_IF_ERROR(LookupNode(consumer, &n));
    Node* node_ptr = n->node.get();

    // check if this value has the same producer already
    if (node_ptr == v->producer) {
      return InvalidArgumentError("Node is a producer of the value");
    }

    // check if this value has the same consumer already
    if (IsInput(consumer, value)) {
      return InvalidArgumentError("Node is already a consumer of the value");
    }

    n->inputs.push_back(value_ptr);
    v->consumers.push_back(node_ptr);
    return OkStatus();
  }

  Status RemoveConsumer(NodeId consumer, ValueId value) final {
    ValueDef* v;
    RETURN_IF_ERROR(LookupValue(value, &v));
    Value<TensorT>* value_ptr = v->value.get();
    NodeDef* n;
    RETURN_IF_ERROR(LookupNode(consumer, &n));
    Node* node_ptr = n->node.get();
    if (!IsInput(consumer, value)) {
      return InvalidArgumentError("Node is not a consumer of the value");
    }
    Erase(&n->inputs, value_ptr);
    Erase(&v->consumers, node_ptr);
    return OkStatus();
  }

  Status DeleteNode(NodeId id) final {
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
    return OkStatus();
  }

  Status DeleteValue(ValueId id) final {
    ValueDef* v;
    RETURN_IF_ERROR(LookupValue(id, &v));
    Value<TensorT>* value_ptr = v->value.get();
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
    return OkStatus();
  }

  Status MakeExactCopy(Model<TensorT>* model) const {
    model->nodes_.clear();
    model->values_.clear();
    model->name_ = name_;
    for (auto& value_def : values_) {
      model->values_.push_back({});
      if (value_def.value) {
        model->values_.back().value =
            absl::make_unique<Value<TensorT>>(*value_def.value);
      }
    }
    for (auto& node_def : nodes_) {
      model->nodes_.push_back({});
      if (node_def.node) {
        model->nodes_.back().node = absl::make_unique<Node>(*node_def.node);
        for (auto output : node_def.outputs) {
          RETURN_IF_ERROR(model->SetProducer(node_def.node->id, output->id));
        }
        for (auto input : node_def.inputs) {
          RETURN_IF_ERROR(model->AddConsumer(node_def.node->id, input->id));
        }
      }
    }
    return OkStatus();
  }

 private:
  struct NodeDef {
    std::vector<Value<TensorT>*> inputs;
    std::vector<Value<TensorT>*> outputs;
    std::unique_ptr<Node> node;
  };

  struct ValueDef {
    Node* producer = nullptr;
    std::vector<Node*> consumers;
    std::unique_ptr<Value<TensorT>> value;
  };

  bool IsInput(NodeId node, ValueId value) {
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

  template <typename T>
  static void Erase(std::vector<T>* values, T value) {
    values->erase(std::find(values->begin(), values->end(), value));
  }

  // @return non-nullptr NodeDef that has valid Node or an error
  Status LookupNode(NodeId id, NodeDef** node_def) {
    if (id >= nodes_.size()) {
      return OutOfRangeError("NodeId is out of range");
    }
    auto& n = nodes_[id];
    if (!n.node) {
      return OutOfRangeError("Node is already deleted");
    }
    *node_def = &n;
    return OkStatus();
  }

  // @return non-nullptr ValueDef that has valid Value or an error
  Status LookupValue(ValueId id, ValueDef** value_def) {
    if (id >= values_.size()) {
      return OutOfRangeError("ValueId is out of range");
    }
    auto& v = values_[id];
    if (!v.value) {
      return OutOfRangeError("Value is already deleted");
    }
    *value_def = &v;
    return OkStatus();
  }

  template <typename Pred>
  std::vector<Value<TensorT>*> FilterValues(const Pred& predicate) const {
    std::vector<Value<TensorT>*> values;
    values.reserve(values_.size());
    for (auto& v : values_) {
      if (v.value != nullptr && predicate(v)) {
        values.push_back(v.value.get());
      }
    }
    return values;
  }

  template <typename Pred>
  std::vector<Node*> FilterNodes(const Pred& predicate) const {
    std::vector<Node*> nodes;
    nodes.reserve(nodes_.size());
    for (auto& n : nodes_) {
      if (n.node != nullptr && predicate(n)) {
        nodes.push_back(n.node.get());
      }
    }
    return nodes;
  }

  std::string name_;

  // There are two approaches possible: wrap entire NodeDef and ValueDef into
  // unique_ptr and store it in values_ and nodes_ or store it by value.
  // We store it by value here to make introspection calls cheaper.
  std::vector<ValueDef> values_;
  std::vector<NodeDef> nodes_;
};

// Removes to_remove node that precedes to_keep node only if to_remove has
// outputs that are consumed only by to_keep. In such case to_keep inherits all
// to_remove inputs.
template <typename TensorT>
Status RemovePrecedingNode(Graph<TensorT>* graph, const Node* to_remove,
                           const Node* to_keep) {
  // Make sure all outputs from to_remove are consumed by to_keep.
  for (auto output : graph->FindOutputs(to_remove->id)) {
    auto consumers = graph->FindConsumers(output->id);
    if (consumers.size() > 1 ||
        (consumers.size() == 1 && consumers[0] != to_keep)) {
      return InvalidArgumentError(
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

// Removes to_remove node that follows to_keep node only if to_remove has inputs
// that are produced by to_keep. to_keep inherits all to_remove inputs.
template <typename TensorT>
Status RemoveFollowingNode(Graph<TensorT>* graph, const Node* to_remove,
                           const Node* to_keep) {
  // Make sure all inputs to to_remove are produced by to_keep.
  for (auto input : graph->FindInputs(to_remove->id)) {
    Node* producer = graph->FindProducer(input->id);
    if (producer->id != to_keep->id) {
      return InvalidArgumentError("To_remove node has other inputs");
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

// Removes to_remove node.
// Requires that node has one input and one output;
template <typename TensorT>
Status RemoveOneInputOneOutputNode(Graph<TensorT>* graph,
                                   const Node* to_remove) {
  auto inputs = graph->FindInputs(to_remove->id);
  auto outputs = graph->FindOutputs(to_remove->id);
  if (inputs.size() != 1 || outputs.size() != 1) {
    return InvalidArgumentError(
        "To_remove node must have 1 input and 1 output");
  }
  auto input_id = inputs[0]->id;
  auto output_id = outputs[0]->id;
  Node* producer = graph->FindProducer(input_id);
  auto consumers = graph->FindConsumers(output_id);
  RETURN_IF_ERROR(graph->DeleteNode(to_remove->id));
  for (auto& consumer : consumers) {
    RETURN_IF_ERROR(graph->ReplaceInput(consumer->id, output_id, input_id));
  }
  RETURN_IF_ERROR(graph->DeleteValue(output_id));
  if (!producer && consumers.empty()) {
    RETURN_IF_ERROR(graph->DeleteValue(input_id));
  }
  return OkStatus();
}

template <typename TensorT>
Status AddOutput(Graph<TensorT>* graph, const Node* from_node,
                 Value<TensorT>** output) {
  auto link = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(from_node->id, link->id));
  *output = link;
  return OkStatus();
}

template <typename TensorT>
Status ConnectTwoNodes(Graph<TensorT>* graph, const Node* from_node,
                       const Node* to_node, Value<TensorT>** output) {
  Value<TensorT>* link;
  RETURN_IF_ERROR(AddOutput(graph, from_node, &link));
  RETURN_IF_ERROR(graph->AddConsumer(to_node->id, link->id));
  *output = link;
  return OkStatus();
}

using GraphFloat32 = Model<TensorRef<BHWC>>;

// @return true if all tensors have same batch value.
inline bool IsBatchMatchesForAllValues(const GraphFloat32& model) {
  const int32_t b = model.values()[0]->tensor.shape.b;
  for (auto value : model.values()) {
    if (value->tensor.shape.b != b) {
      return false;
    }
  }
  return true;
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
