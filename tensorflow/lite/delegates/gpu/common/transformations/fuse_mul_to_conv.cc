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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

class MergeConvolutionWithMul : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& conv_node = *sequence[0];
    auto& mul_node = *sequence[1];
    if (mul_node.operation.type != ToString(OperationType::MUL) &&
        mul_node.operation.type != ToString(OperationType::MULTIPLY_SCALAR)) {
      return {TransformStatus::SKIPPED, ""};
    }

    MultiplyScalarAttributes mul_attr =
        absl::any_cast<MultiplyScalarAttributes>(mul_node.operation.attributes);
    if (!absl::get_if<Tensor<Linear, DataType::FLOAT32>>(
            &mul_attr.param) &&
        !absl::get_if<float>(&mul_attr.param)) {
      return {
          TransformStatus::DECLINED,
          "This fuse applicable only for broadcast or scalar multiplication."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseConvolution2DWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseConvolutionTransposedWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseDepthwiseConvolution2DWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseFullyConnectedWithMultiply(mul_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    Status status = RemoveFollowingNode(graph, &mul_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove mul node after convolution: " +
                  status.error_message()};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

class MergeMulWithConvolution : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& conv_node = *sequence[1];
    auto& mul_node = *sequence[0];
    if (mul_node.operation.type != ToString(OperationType::MUL) &&
        mul_node.operation.type != ToString(OperationType::MULTIPLY_SCALAR)) {
      return {TransformStatus::SKIPPED, ""};
    }

    MultiplyScalarAttributes mul_attr =
        absl::any_cast<MultiplyScalarAttributes>(mul_node.operation.attributes);
    if (!absl::get_if<Tensor<Linear, DataType::FLOAT32>>(
            &mul_attr.param) &&
        !absl::get_if<float>(&mul_attr.param)) {
      return {
          TransformStatus::DECLINED,
          "This fuse applicable only for broadcast or scalar multiplication."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithConvolution2D(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithConvolutionTransposed(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithDepthwiseConvolution2D(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithFullyConnected(mul_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    Status status = RemovePrecedingNode(graph, &mul_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove mul node after convolution: " +
                  status.error_message()};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithMul() {
  return absl::make_unique<MergeConvolutionWithMul>();
}

std::unique_ptr<SequenceTransformation> NewMergeMulWithConvolution() {
  return absl::make_unique<MergeMulWithConvolution>();
}

void FuseConvolution2DWithMultiply(const MultiplyScalarAttributes& mul_attr,
                                   Convolution2DAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({d, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseDepthwiseConvolution2DWithMultiply(
    const MultiplyScalarAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int g = 0; g < attr->weights.shape.o; ++g) {
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const int d = s * attr->weights.shape.o + g;
      const float multiplier = mul ? mul->data[d] : *mul_scalar;
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({g, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
      if (!attr->bias.data.empty()) {
        attr->bias.data[d] *= multiplier;
      }
    }
  }
}

void FuseConvolutionTransposedWithMultiply(
    const MultiplyScalarAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({d, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseFullyConnectedWithMultiply(const MultiplyScalarAttributes& mul_attr,
                                    FullyConnectedAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const int index = attr->weights.shape.LinearIndex({d, 0, 0, s});
      attr->weights.data[index] *= multiplier;
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseMultiplyWithConvolution2D(const MultiplyScalarAttributes& mul_attr,
                                   Convolution2DAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({d, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithDepthwiseConvolution2D(
    const MultiplyScalarAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int g = 0; g < attr->weights.shape.o; ++g) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({g, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithConvolutionTransposed(
    const MultiplyScalarAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({d, k_y, k_x, s});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithFullyConnected(const MultiplyScalarAttributes& mul_attr,
                                    FullyConnectedAttributes* attr) {
  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      const int index = attr->weights.shape.LinearIndex({d, 0, 0, s});
      attr->weights.data[index] *= multiplier;
    }
  }
}

}  // namespace gpu
}  // namespace tflite
