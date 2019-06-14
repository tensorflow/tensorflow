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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

template <typename T>
TransformResult FillBias(Node* node) {
  auto& attr = absl::any_cast<T&>(node->operation.attributes);
  if (attr.bias.data.empty()) {
    const int dst_channels = attr.weights.shape.o;
    attr.bias = MakeZeroTensor<Linear, DataType::FLOAT32>(Linear(dst_channels));
    return {TransformStatus::APPLIED, "Added bias"};
  }
  return {TransformStatus::SKIPPED, ""};
}

template TransformResult FillBias<Convolution2DAttributes>(Node* node);
template TransformResult FillBias<ConvolutionTransposedAttributes>(Node* node);
template TransformResult FillBias<DepthwiseConvolution2DAttributes>(Node* node);
template TransformResult FillBias<FullyConnectedAttributes>(Node* node);

class AddBias : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      return FillBias<Convolution2DAttributes>(node);
    }
    if (node->operation.type ==
        ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      return FillBias<ConvolutionTransposedAttributes>(node);
    }
    if (node->operation.type ==
        ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      return FillBias<DepthwiseConvolution2DAttributes>(node);
    }
    if (node->operation.type == ToString(OperationType::FULLY_CONNECTED)) {
      return FillBias<FullyConnectedAttributes>(node);
    }
    return {TransformStatus::SKIPPED, ""};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewAddBias() {
  return absl::make_unique<AddBias>();
}

}  // namespace gpu
}  // namespace tflite
