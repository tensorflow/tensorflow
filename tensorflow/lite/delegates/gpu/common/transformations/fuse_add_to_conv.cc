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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

void FuseBiasWithAddAttributes(const AddAttributes& add_attr,
                               const int channels,
                               Tensor<Linear, DataType::FLOAT32>* bias) {
  auto add = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
  auto add_scalar = absl::get_if<float>(&add_attr.param);
  if (bias->data.empty()) {
    *bias = MakeZeroTensor<Linear, DataType::FLOAT32>(Linear(channels));
  }
  for (int d = 0; d < channels; ++d) {
    bias->data[d] += add ? add->data[d] : *add_scalar;
  }
}

class MergeConvolutionWithAdd : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& conv_node = *sequence[0];
    auto& add_node = *sequence[1];
    if (add_node.operation.type != ToString(OperationType::ADD)) {
      return {TransformStatus::SKIPPED, ""};
    }
    AddAttributes add_attr =
        absl::any_cast<AddAttributes>(add_node.operation.attributes);
    if (!absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param) &&
        !absl::get_if<float>(&add_attr.param)) {
      return {TransformStatus::DECLINED,
              "This fuse applicable only for broadcast or scalar addition."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseConvolution2DWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseConvolutionTransposedWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseDepthwiseConvolution2DWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseFullyConnectedWithAdd(add_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    Status status = RemoveFollowingNode(graph, &add_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove add node after convolution: " +
                  status.error_message()};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

class MergeAddWithConvolution : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    auto& conv_node = *sequence[1];
    auto& add_node = *sequence[0];
    if (add_node.operation.type != ToString(OperationType::ADD)) {
      return {TransformStatus::SKIPPED, ""};
    }
    AddAttributes add_attr =
        absl::any_cast<AddAttributes>(add_node.operation.attributes);
    if (!absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param) &&
        !absl::get_if<float>(&add_attr.param)) {
      return {TransformStatus::DECLINED,
              "This fuse applicable only for broadcast or scalar addition."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseAddWithConvolution2D(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseAddWithDepthwiseConvolution2D(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseAddWithFullyConnected(add_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    Status status = RemovePrecedingNode(graph, &add_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove add node after convolution: " +
                  status.error_message()};
    }
    return {TransformStatus::APPLIED, ""};
  }
};
}  // namespace

std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithAdd() {
  return absl::make_unique<MergeConvolutionWithAdd>();
}

std::unique_ptr<SequenceTransformation> NewMergeAddWithConvolution() {
  return absl::make_unique<MergeAddWithConvolution>();
}

void FuseConvolution2DWithAdd(const AddAttributes& add_attr,
                              Convolution2DAttributes* attr) {
  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

void FuseDepthwiseConvolution2DWithAdd(const AddAttributes& add_attr,
                                       DepthwiseConvolution2DAttributes* attr) {
  FuseBiasWithAddAttributes(
      add_attr, attr->weights.shape.o * attr->weights.shape.i, &attr->bias);
}

void FuseConvolutionTransposedWithAdd(const AddAttributes& add_attr,
                                      ConvolutionTransposedAttributes* attr) {
  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

void FuseFullyConnectedWithAdd(const AddAttributes& add_attr,
                               FullyConnectedAttributes* attr) {
  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

void FuseAddWithConvolution2D(const AddAttributes& add_attr,
                              Convolution2DAttributes* attr) {
  auto add = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
  auto add_scalar = absl::get_if<float>(&add_attr.param);
  if (attr->bias.data.empty()) {
    attr->bias = MakeZeroTensor<Linear, DataType::FLOAT32>(
        Linear(attr->weights.shape.o));
  }
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const float add_value = add ? add->data[s] : *add_scalar;
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{d, k_y, k_x, s}});
          attr->bias.data[d] += attr->weights.data[index] * add_value;
        }
      }
    }
  }
}

void FuseAddWithDepthwiseConvolution2D(const AddAttributes& add_attr,
                                       DepthwiseConvolution2DAttributes* attr) {
  auto add = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
  auto add_scalar = absl::get_if<float>(&add_attr.param);
  if (attr->bias.data.empty()) {
    attr->bias = MakeZeroTensor<Linear, DataType::FLOAT32>(
        Linear(attr->weights.shape.o * attr->weights.shape.i));
  }
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float add_value = add ? add->data[s] : *add_scalar;
    for (int g = 0; g < attr->weights.shape.o; ++g) {
      const int d = s * attr->weights.shape.o + g;
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{g, k_y, k_x, s}});
          attr->bias.data[d] += attr->weights.data[index] * add_value;
        }
      }
    }
  }
}

void FuseAddWithFullyConnected(const AddAttributes& add_attr,
                               FullyConnectedAttributes* attr) {
  auto add = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
  auto add_scalar = absl::get_if<float>(&add_attr.param);
  if (attr->bias.data.empty()) {
    attr->bias = MakeZeroTensor<Linear, DataType::FLOAT32>(
        Linear(attr->weights.shape.o));
  }
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const float add_value = add ? add->data[s] : *add_scalar;
      const int index = attr->weights.shape.LinearIndex({{d, 0, 0, s}});
      attr->bias.data[d] += attr->weights.data[index] * add_value;
    }
  }
}

}  // namespace gpu
}  // namespace tflite
