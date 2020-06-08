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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "absl/types/optional.h"
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

// Used to emulate quantized behavior.
struct QuantizationParams {
  float min = 0;
  float max = 0;
  float scale = 0;
};

// Connects tensor's producer and operation that depends on this tensor.
struct Value {
  const ValueId id;
  TensorRef<BHWC> tensor;
  absl::optional<QuantizationParams> quant_params;
};

struct Operation {
  std::string type;
  absl::any attributes;
};

struct Node {
  const NodeId id;
  Operation operation;
};

// A DAG that consists of nodes and values. Each value may have a single
// producer node and multiple consumer nodes. Therefore, each node may have
// multiple input and output values.
//
// Value that does not have a producer is a graph's input. Value that does not
// have a consumer is a graph's output.
//
// It keeps values and nodes referenced by their index in a vector. Therefore,
// nodes and values are never deleted, but rather erased, where corresponding
// index remains.
//
// It is possible to re-use removed indices, but it is not implemented yet.
class GraphFloat32 {
 public:
  // @return a collection of nodes in this graph.
  std::vector<Node*> nodes() const;

  // @return a collection of values in this graph.
  std::vector<Value*> values() const;

  // @return graph inputs, that are values without producers.
  std::vector<Value*> inputs() const;

  // @return graph outputs, that are values without consumers.
  std::vector<Value*> outputs() const;

  // @return inputs into the given node. Returns empty vector for deleted node.
  std::vector<Value*> FindInputs(NodeId id) const;

  // @return outputs from the given node. Returns empty vector for deleted node.
  std::vector<Value*> FindOutputs(NodeId id) const;

  bool IsGraphInput(ValueId id) const;

  bool IsGraphOutput(ValueId id) const;

  // @return producer of the given value. Returns nullptr for deleted value.
  Node* FindProducer(ValueId id) const;

  // @return consumers of the given value. Returns empty vector for deleted
  // value.
  std::vector<Node*> FindConsumers(ValueId id) const;

  // @return a node or nullptr if node with the given id is not present.
  Node* GetNode(NodeId id) const;

  // @return a value or nullptr if value with the given id is not present.
  Value* GetValue(ValueId id) const;

  //////////////////////////////////////////////////////////////////////////////
  // Graph manipulation functions are below
  //////////////////////////////////////////////////////////////////////////////

  // @return new node created in this graph
  // NOTE: nodes should be created in the topological order, e.g. node A that
  // depends on a value from node B should be created after node B.
  Node* NewNode();

  // Insert Node after another in the execution plan.
  absl::Status InsertNodeAfter(NodeId id, Node** new_node);

  // @return new value created in this graph
  Value* NewValue();

  // Sets a producer for the given value. There could be a single producer
  // for a value. If a value had another producer, it will reassign producer
  // appropriately. If a value didn't have a producer, it will be removed
  // from a graph's input.
  absl::Status SetProducer(NodeId producer, ValueId value);

  // Removes a producer for the given value. Value becomes producer-less and
  // therefore becomes graph's input.
  absl::Status RemoveProducer(ValueId value);

  // Sets a consumer for the given value. There could be multiple consumers
  // for a value.
  absl::Status AddConsumer(NodeId consumer, ValueId value);

  // Replace input value for given node.
  absl::Status ReplaceInput(NodeId node, ValueId old_value, ValueId new_value);

  // Removes a consumer for the given value. If value does not have any
  // consumers it becomes graph's output.
  absl::Status RemoveConsumer(NodeId consumer, ValueId value);

  // Removes node from this graph. For all input values this node will be
  // removed from consumers and for all output values a producer will be
  // removed.
  absl::Status DeleteNode(NodeId id);

  // Removes value from this graph. It will be removed from inputs for all
  // dependent nodes. A node that was a producer of this value will loose its
  // output.
  absl::Status DeleteValue(ValueId id);

  absl::Status MakeExactCopy(GraphFloat32* model) const;

 private:
  struct NodeDef {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::unique_ptr<Node> node;
  };

  struct ValueDef {
    Node* producer = nullptr;
    std::vector<Node*> consumers;
    std::unique_ptr<Value> value;
  };

  bool IsInput(NodeId node, ValueId value);

  template <typename T>
  static void Erase(std::vector<T>* values, T value) {
    values->erase(std::find(values->begin(), values->end(), value));
  }

  // @return non-nullptr NodeDef that has valid Node or an error
  absl::Status LookupNode(NodeId id, NodeDef** node_def);

  // @return non-nullptr ValueDef that has valid Value or an error
  absl::Status LookupValue(ValueId id, ValueDef** value_def);

  template <typename Pred>
  std::vector<Value*> FilterValues(const Pred& predicate) const {
    std::vector<Value*> values;
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
    for (const auto id : execution_plan_) {
      auto& n = nodes_.at(id);
      if (n.node != nullptr && predicate(n)) {
        nodes.push_back(n.node.get());
      }
    }
    return nodes;
  }

  // There are two approaches possible: wrap entire NodeDef and ValueDef into
  // unique_ptr and store it in values_ and nodes_ or store it by value.
  // We store it by value here to make introspection calls cheaper.
  std::vector<ValueDef> values_;

  std::map<NodeId, NodeDef> nodes_;
  // Node Ids in order of execution.
  std::vector<NodeId> execution_plan_;
};

// Removes to_remove node that precedes to_keep node only if to_remove has
// outputs that are consumed only by to_keep. In such case to_keep inherits all
// to_remove inputs.
absl::Status RemovePrecedingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep);

// Removes to_remove node that follows to_keep node only if to_remove has inputs
// that are produced by to_keep. to_keep inherits all to_remove inputs.
absl::Status RemoveFollowingNode(GraphFloat32* graph, const Node* to_remove,
                                 const Node* to_keep);

// Removes to_remove node.
// Requires that node has one input and one output;
absl::Status RemoveOneInputOneOutputNode(GraphFloat32* graph,
                                         const Node* to_remove);

absl::Status AddOutput(GraphFloat32* graph, const Node* from_node,
                       Value** output);

absl::Status ConnectTwoNodes(GraphFloat32* graph, const Node* from_node,
                             const Node* to_node, Value** output);

// @return true if all tensors have same batch value.
bool IsBatchMatchesForAllValues(const GraphFloat32& model);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_H_
