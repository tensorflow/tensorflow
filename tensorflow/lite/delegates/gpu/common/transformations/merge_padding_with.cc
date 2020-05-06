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

#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/matching.h"

namespace tflite {
namespace gpu {
namespace {

template <typename Attr>
class MergePaddingWith2DOperation : public SequenceTransformation {
 public:
  explicit MergePaddingWith2DOperation(OperationType operation_type)
      : operations_to_match_(
            {ToString(OperationType::PAD), ToString(operation_type)}) {}

  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    if (!MatchesByOperationType(sequence, operations_to_match_)) {
      return {TransformStatus::SKIPPED, ""};
    }

    Node* pad_node = sequence.front();
    Node* op_node = sequence.back();

    PadAttributes pad_attr =
        absl::any_cast<PadAttributes>(pad_node->operation.attributes);

    if (pad_attr.type != PaddingContentType::ZEROS) {
      return {TransformStatus::DECLINED, "Only Zero padding is supported."};
    }
    if (pad_attr.appended.c != 0 || pad_attr.prepended.c != 0 ||
        pad_attr.appended.b != 0 || pad_attr.prepended.b != 0) {
      return {TransformStatus::DECLINED,
              "Pad has non-zero padding on non HW axis."};
    }

    Attr* node_attr = absl::any_cast<Attr>(&op_node->operation.attributes);
    absl::Status status = RemovePrecedingNode(graph, pad_node, op_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove Pad node with Operation node: " +
                  std::string(status.message())};
    }

    node_attr->padding.appended.h += pad_attr.appended.h;
    node_attr->padding.appended.w += pad_attr.appended.w;
    node_attr->padding.prepended.h += pad_attr.prepended.h;
    node_attr->padding.prepended.w += pad_attr.prepended.w;
    return {
        TransformStatus::APPLIED,
        absl::StrCat("Added padding: prepended = {h = ", pad_attr.prepended.h,
                     ", w = ", pad_attr.prepended.w, "}, appended = { h = ",
                     pad_attr.appended.h, ", w = ", pad_attr.appended.w, "}")};
  }

 private:
  const std::vector<std::string> operations_to_match_;
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMergePaddingWithPooling() {
  return absl::make_unique<MergePaddingWith2DOperation<Pooling2DAttributes>>(
      OperationType::POOLING_2D);
}

std::unique_ptr<SequenceTransformation> NewMergePaddingWithConvolution2D() {
  return absl::make_unique<
      MergePaddingWith2DOperation<Convolution2DAttributes>>(
      OperationType::CONVOLUTION_2D);
}

std::unique_ptr<SequenceTransformation>
NewMergePaddingWithDepthwiseConvolution() {
  return absl::make_unique<
      MergePaddingWith2DOperation<DepthwiseConvolution2DAttributes>>(
      OperationType::DEPTHWISE_CONVOLUTION);
}

class MergePaddingWithAddOperation : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::PAD)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    if (input_shape.c % 4 != 0) {
      return {TransformStatus::DECLINED,
              "Pad with input where src_channels % 4 != 0"};
    }

    PadAttributes pad_attr =
        absl::any_cast<PadAttributes>(node->operation.attributes);

    if (pad_attr.type != PaddingContentType::ZEROS) {
      return {TransformStatus::DECLINED, "Only Zero padding is supported."};
    }
    if (pad_attr.prepended != BHWC(0, 0, 0, 0) || pad_attr.appended.h != 0 ||
        pad_attr.appended.w != 0 || pad_attr.appended.b != 0) {
      return {TransformStatus::DECLINED,
              "Pad has padding not only in appended channels axis."};
    }

    auto pad_output = graph->FindOutputs(node->id)[0];
    auto consumer_nodes = graph->FindConsumers(pad_output->id);
    if (consumer_nodes.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto add_node = consumer_nodes[0];
    auto consumer_type = OperationTypeFromString(add_node->operation.type);
    if (consumer_type != OperationType::ADD) {
      return {TransformStatus::SKIPPED, ""};
    }

    AddAttributes add_attr =
        absl::any_cast<AddAttributes>(add_node->operation.attributes);
    const auto add_broadcast =
        absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
    const float* add_scalar = absl::get_if<float>(&add_attr.param);
    if (add_broadcast || add_scalar) {
      return {TransformStatus::SKIPPED,
              "Cannot remove padding when this broadcast/scalar ADD"};
    }

    absl::Status status = RemovePrecedingNode(graph, node, add_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove Pad node " + std::string(status.message())};
    }

    return {TransformStatus::APPLIED,
            "Removed padding with zeroes in appended channels dimension"};
  }
};

std::unique_ptr<NodeTransformation> NewMergePaddingWithAdd() {
  return absl::make_unique<MergePaddingWithAddOperation>();
}

}  // namespace gpu
}  // namespace tflite
