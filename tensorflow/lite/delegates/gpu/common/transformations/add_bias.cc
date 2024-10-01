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

#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"

#include <any>
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

TransformResult FillBias(
    int output_channels,
    tflite::gpu::Tensor<Linear, DataType::FLOAT32>* biases) {
  if (biases->data.empty()) {
    *biases =
        MakeZeroTensor<Linear, DataType::FLOAT32>(Linear(output_channels));
    return {TransformStatus::APPLIED, "Added bias"};
  }
  if (biases->shape.v != output_channels) {
    float last_value = biases->data.back();
    biases->shape.v = output_channels;
    biases->data.resize(output_channels, last_value);
    return {TransformStatus::APPLIED, "Bias extended"};
  }
  return {TransformStatus::SKIPPED, ""};
}

class AddBias : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      if (graph->FindInputs(node->id).size() != 1) {
        return {TransformStatus::DECLINED,
                "This transformation is only applicable to conv with one "
                "runtime input."};
      }
      auto& attr =
          std::any_cast<Convolution2DAttributes&>(node->operation.attributes);
      return FillBias(attr.weights.shape.o, &attr.bias);
    }
    if (node->operation.type ==
        ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      auto& attr = std::any_cast<ConvolutionTransposedAttributes&>(
          node->operation.attributes);
      return FillBias(attr.weights.shape.o, &attr.bias);
    }
    if (node->operation.type ==
        ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      if (graph->FindInputs(node->id).size() != 1) {
        return {TransformStatus::DECLINED,
                "This transformation is only applicable to depth wise conv "
                "with one "
                "runtime input."};
      }
      auto& attr = std::any_cast<DepthwiseConvolution2DAttributes&>(
          node->operation.attributes);
      return FillBias(attr.weights.shape.o * attr.weights.shape.i, &attr.bias);
    }
    if (node->operation.type == ToString(OperationType::FULLY_CONNECTED)) {
      auto& attr =
          std::any_cast<FullyConnectedAttributes&>(node->operation.attributes);
      return FillBias(attr.weights.shape.o, &attr.bias);
    }
    if (node->operation.type == ToString(OperationType::FULLY_CONNECTED_INT8)) {
      auto& attr = std::any_cast<FullyConnectedInt8Attributes&>(
          node->operation.attributes);
      return FillBias(attr.weights.shape.o, &attr.bias);
    }
    return {TransformStatus::SKIPPED, ""};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewAddBias() {
  return std::make_unique<AddBias>();
}

}  // namespace gpu
}  // namespace tflite
