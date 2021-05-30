/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/transformations/merge_densify.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

class MergeDensify : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    // Only CONV_2D & DEPTHWISE_CONV_2D.
    const std::string& node_type = node->operation.type;
    if (node_type != ToString(OperationType::CONVOLUTION_2D) &&
        node_type != ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      return {TransformStatus::SKIPPED, ""};
    }

    // Only with a runtime weights.
    const std::vector<Value*> inputs = graph->FindInputs(node->id);
    if (inputs.size() != 2) return {TransformStatus::SKIPPED, ""};

    const Node* dequantize_or_densify = graph->FindProducer(inputs[1]->id);
    if (!dequantize_or_densify ||
        (dequantize_or_densify->operation.type !=
             ToString(OperationType::DENSIFY) &&
         dequantize_or_densify->operation.type !=
             ToString(OperationType::QUANTIZE_AND_DEQUANTIZE))) {
      return {TransformStatus::SKIPPED, ""};
    }
    const Node* dequantize_node;
    const Node* densify_node;
    if (dequantize_or_densify->operation.type ==
        ToString(OperationType::QUANTIZE_AND_DEQUANTIZE)) {
      dequantize_node = dequantize_or_densify;
      densify_node =
          graph->FindProducer(graph->FindInputs(dequantize_node->id)[0]->id);
      if (!densify_node ||
          densify_node->operation.type != ToString(OperationType::DENSIFY)) {
        return {TransformStatus::SKIPPED, ""};
      }
    } else {
      dequantize_node = nullptr;
      densify_node = dequantize_or_densify;
    }

    // Create a copy of the const tensor with a cast from BHWC to OHWI.
    const Tensor<BHWC, DataType::FLOAT32>& src =
        absl::any_cast<DensifyAttributes>(&densify_node->operation.attributes)
            ->tensor;
    Tensor<OHWI, DataType::FLOAT32> dst;
    dst.id = src.id;
    dst.shape = OHWI(src.shape.b, src.shape.h, src.shape.w, src.shape.c);
    dst.data = src.data;

    // Remove DEQUANTIZE.
    if (dequantize_node) {
      const auto status = RemovePrecedingNode(graph, dequantize_node, node);
      if (!status.ok()) return {TransformStatus::INVALID, status.ToString()};
    }

    // Remove DENSIFY.
    const auto status = RemovePrecedingNode(graph, densify_node, node);
    if (!status.ok()) return {TransformStatus::INVALID, status.ToString()};

    // Update CONV_2D / DEPTHWISE_CONV_2D weights.
    if (node->operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      absl::any_cast<Convolution2DAttributes>(&node->operation.attributes)
          ->weights = std::move(dst);
    } else {
      absl::any_cast<DepthwiseConvolution2DAttributes>(
          &node->operation.attributes)
          ->weights = std::move(dst);
    }
    return {TransformStatus::APPLIED, ""};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewMergeDensify() {
  return absl::make_unique<MergeDensify>();
}

}  // namespace gpu
}  // namespace tflite
