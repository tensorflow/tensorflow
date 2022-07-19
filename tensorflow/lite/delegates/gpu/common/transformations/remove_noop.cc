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

#include "tensorflow/lite/delegates/gpu/common/transformations/remove_noop.h"

#include <algorithm>
#include <any>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

using ShouldRemoveOperation = std::function<bool(GraphFloat32* graph, Node*)>;

class RemoveOperation : public SequenceTransformation {
 public:
  explicit RemoveOperation(ShouldRemoveOperation remove_predicate)
      : remove_predicate_(std::move(remove_predicate)) {}

  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    Node* prev_op_node = sequence.front();
    Node* op_node = sequence.back();
    if (!remove_predicate_(graph, op_node)) {
      return {TransformStatus::SKIPPED, ""};
    }
    absl::Status status = RemoveFollowingNode(graph, op_node, prev_op_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }

 private:
  ShouldRemoveOperation remove_predicate_;
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputConcat() {
  // Using SequenceTransformation implies that CONCAT has a single input.
  auto type = ToString(OperationType::CONCAT);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        return type == node->operation.type;
      });
}

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputAdd() {
  // Using SequenceTransformation implies that ADD has a single input.
  auto type = ToString(OperationType::ADD);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        if (node->operation.type != type) {
          return false;
        }
        auto& attr = absl::any_cast<const ElementwiseAttributes&>(
            node->operation.attributes);
        return !absl::holds_alternative<Tensor<HWC, DataType::FLOAT32>>(
                   attr.param) &&
               !absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
                   attr.param) &&
               !absl::holds_alternative<float>(attr.param);
      });
}

std::unique_ptr<SequenceTransformation> NewRemoveDegenerateUpsampling() {
  auto type = ToString(OperationType::RESIZE);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        if (node->operation.type != type) {
          return false;
        }
        auto inputs = graph->FindInputs(node->id);
        auto outputs = graph->FindOutputs(node->id);
        return inputs.size() == 1 && outputs.size() == 1 &&
               inputs[0]->tensor.shape == outputs[0]->tensor.shape;
      });
}

class RemoveIdentityReshape : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::RESHAPE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    const auto& reshape_attr =
        absl::any_cast<const ReshapeAttributes&>(node->operation.attributes);
    if (input_shape != reshape_attr.new_shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto output = graph->FindOutputs(node->id)[0];
    const auto& graph_outputs = graph->outputs();
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end()) {
      return {TransformStatus::SKIPPED,
              "Can not apply transformation when node output is graph output"};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED,
            "Removed reshape with input_shape == output_shape."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityReshape() {
  return absl::make_unique<RemoveIdentityReshape>();
}

class RemoveIdentityStridedSlice : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::SLICE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto input = graph->FindInputs(node->id)[0];
    auto output = graph->FindOutputs(node->id)[0];
    const auto& slice_attr =
        absl::any_cast<const SliceAttributes&>(node->operation.attributes);
    if (input->tensor.shape != output->tensor.shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.starts != BHWC(0, 0, 0, 0)) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.strides != BHWC(1, 1, 1, 1)) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.ends != output->tensor.shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    const auto& graph_outputs = graph->outputs();
    const auto& graph_inputs = graph->inputs();
    const bool input_is_graph_input =
        std::find(graph_inputs.begin(), graph_inputs.end(), input) !=
        graph_inputs.end();
    const bool output_is_graph_output =
        std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end();
    if (input_is_graph_input && output_is_graph_output) {
      return {TransformStatus::SKIPPED,
              "Can not apply transformation when node input is graph input and "
              "node output is graph output"};
    }
    if (output_is_graph_output) {
      if (graph->FindConsumers(input->id).size() != 1) {
        return {TransformStatus::SKIPPED,
                "Can not apply transformation when node output is graph output "
                "and input consumed by other nodes."};
      }
      absl::Status status = RemoveSimpleNodeKeepOutput(graph, node);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                "Unable to remove a node: " + std::string(status.message())};
      }
      return {TransformStatus::APPLIED, "Removed identity strided slice."};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED, "Removed identity strided slice."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityStridedSlice() {
  return absl::make_unique<RemoveIdentityStridedSlice>();
}

class RemoveCastAfterLogicalOp : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& logical_node = *sequence[0];
    auto op_type = OperationTypeFromString(logical_node.operation.type);
    if (!(op_type == OperationType::GREATER ||
          op_type == OperationType::GREATER_EQUAL ||
          op_type == OperationType::LESS ||
          op_type == OperationType::LESS_EQUAL ||
          op_type == OperationType::EQUAL ||
          op_type == OperationType::NOT_EQUAL)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto& cast_node = *sequence[1];
    if (cast_node.operation.type != ToString(OperationType::CAST)) {
      return {TransformStatus::SKIPPED, ""};
    }

    absl::Status status = RemoveFollowingNode(graph, &cast_node, &logical_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove cast node after logigal node: " +
                  std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

std::unique_ptr<SequenceTransformation> NewRemoveCastAfterLogicalOp() {
  return std::make_unique<RemoveCastAfterLogicalOp>();
}

}  // namespace gpu
}  // namespace tflite
