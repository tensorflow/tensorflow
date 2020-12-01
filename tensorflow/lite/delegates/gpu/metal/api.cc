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

#include "tensorflow/lite/delegates/gpu/metal/api.h"

#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/concat.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/custom_registry.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/padding.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/pooling.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/prelu.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/quantize_and_dequantize.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/relu.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/reshape.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/resize.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/slice.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/softmax.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/space_to_depth.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/transpose_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/winograd.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::vector<ComputeTaskDescriptorPtr> SelectDepthWiseConv(
    ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const metal::RuntimeOptions& options) {
  if (CheckDepthWiseConv3x3Stride1x1Support(attr)) {
    auto gpu_op = DepthWiseConv3x3Stride1x1(input_id, output_id, attr, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else if (CheckDepthWiseConv3x3Stride2Support(attr)) {
    auto gpu_op = DepthWiseConv3x3Stride2(input_id, output_id, attr, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op = DepthWiseConvolution(input_id, output_id, attr, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectConvolutionTransposed(
    ValueId input_id, ValueId output_id,
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const metal::RuntimeOptions& options) {
  if (CheckConvolutionTransposed4x4Support(attr)) {
    auto gpu_op =
        ConvolutionTransposed4x4(input_id, output_id, attr, gpu_info, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op =
        ConvolutionTransposed(input_id, output_id, attr, gpu_info, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectQuantizeAndDequantize(
    ValueId input_id, ValueId output_id,
    const QuantizeAndDequantizeAttributes& attr) {
  auto gpu_op = QuantizeAndDequantize(input_id, output_id, attr);
  return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
}

std::vector<ComputeTaskDescriptorPtr> SelectPReLU(
    const BHWC& src_shape, ValueId input_id, ValueId output_id,
    const PReLUAttributes& attr, const metal::RuntimeOptions& options) {
  auto alpha = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha) {
    auto gpu_op = PReLU(input_id, output_id, attr, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
  auto alpha3d = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha3d) {
    return {};
  }
  if (alpha3d->shape.h != src_shape.h || alpha3d->shape.w != src_shape.w ||
      alpha3d->shape.c != src_shape.c) {
    return {};
  }
  auto gpu_op = PReLUFull(input_id, output_id, attr, options);
  return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
}

std::vector<ComputeTaskDescriptorPtr> SelectReshape(
    const BHWC& src_shape, ValueId input_id, ValueId output_id,
    const ReshapeAttributes& attr) {
  if (src_shape.c % 4 == 0 && attr.new_shape.c % 4 == 0) {
    auto gpu_op = Reshapex4(input_id, output_id, attr);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op = Reshape(input_id, output_id, attr);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectSoftmax(const BHWC& src_shape,
                                                    ValueId input_id,
                                                    ValueId output_id,
                                                    const GpuInfo& gpu_info) {
  if (src_shape.w == 1 && src_shape.h == 1) {
    auto gpu_op = Softmax1x1(input_id, output_id, gpu_info, src_shape.c);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op = Softmax(input_id, output_id, src_shape.c);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectSpaceToDepth(
    ValueId input_id, ValueId output_id, const SpaceToDepthAttributes& attr) {
  auto gpu_op = SpaceToDepth(input_id, output_id, attr);
  return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
}

std::vector<ComputeTaskDescriptorPtr> SelectWinograd4x4To36(
    ValueId input_id, ValueId output_id, const Winograd4x4To36Attributes& attr,
    const GpuInfo& gpu_info, const metal::RuntimeOptions& options) {
  if (gpu_info.IsApple()) {
    auto gpu_op = Winograd4x4To36(input_id, output_id, attr);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op = Winograd4x4To36TileX6(input_id, output_id, attr, options);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectWinograd36To4x4(
    ValueId input_id, ValueId output_id, const Winograd36To4x4Attributes& attr,
    const GpuInfo& gpu_info, const metal::RuntimeOptions& options) {
  if (gpu_info.IsApple()) {
    auto gpu_op = Winograd36To4x4(input_id, output_id, options, attr);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  } else {
    auto gpu_op = Winograd36To4x4Tile4x1(input_id, output_id, options, attr);
    return {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
  }
}

bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr,
                                   const BHWC& dst_shape) {
  const int tiles_x = DivideRoundUp(dst_shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_shape.h, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const bool suitable_attributes =
      attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
      attr.dilations == HW(1, 1) && attr.strides == HW(1, 1);

  const int min_depth = 16;
  const int min_hw = 32;
  const bool recommended_channels =
      src_depth >= min_depth && dst_depth >= min_depth;
  const bool recommended_hw = tiles_x * tiles_y >= min_hw;
  return suitable_attributes && recommended_channels && recommended_hw;
}

absl::Status RegisterPrimaryOps(const GraphFloat32& graph, const Node* node,
                                const std::vector<ValueId>& inputs,
                                const std::vector<ValueId>& outputs,
                                const GpuInfo& gpu_info,
                                const RuntimeOptions& options,
                                int* last_node_id, int* last_value_id,
                                std::map<ValueId, BHWC>* tensor_shapes,
                                std::vector<ComputeTaskDescriptorPtr>* tasks) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return absl::InvalidArgumentError(
        "Only identical batch dimension is supported");
  }
  int node_id = static_cast<int>(node->id);
  auto op_type = OperationTypeFromString(node->operation.type);
  switch (op_type) {
    case OperationType::ADD: {
      if (inputs.size() == 1) {
        if (node->operation.attributes.has_value()) {
          auto attr =
              absl::any_cast<ElementwiseAttributes>(node->operation.attributes);
          auto gpu_op = ElementwiseWithOneInputAndConstantArguent(
              inputs[0], outputs[0], options, op_type, attr.param);
          *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
        } else {
          return absl::UnimplementedError(
              "Missing attributes for single input op: " +
              node->operation.type);
        }
      } else if (inputs.size() == 2) {
        const auto srcs = graph.FindInputs(node_id);
        auto gpu_op = ElementwiseWithTwoInputs(inputs, outputs[0],
                                               srcs[1]->tensor.shape, op_type);
        *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      } else {  // more than 2 inputs
        auto gpu_op = Add(inputs, outputs[0], options);
        *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      }
      break;
    }
    case OperationType::CONCAT: {
      std::vector<BHWC> input_shapes;
      for (auto& input : graph.FindInputs(node->id)) {
        input_shapes.push_back(input->tensor.shape);
      }
      auto gpu_op =
          Concat(inputs, outputs[0],
                 absl::any_cast<ConcatAttributes>(node->operation.attributes),
                 input_shapes);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::CONVOLUTION_2D: {
      if (graph.FindInputs(node->id).size() != 1) {
        return absl::UnimplementedError(
            "Convolution does not support more than 1 runtime tensor");
      }
      const auto src_shape = graph.FindInputs(node_id)[0]->tensor.shape;
      const auto dst_shape = graph.FindOutputs(node_id)[0]->tensor.shape;
      auto attr =
          absl::any_cast<Convolution2DAttributes>(node->operation.attributes);
      if (IsSuitableForWinograd4x4To6x6(attr, dst_shape)) {
        int tiles_x = DivideRoundUp(dst_shape.w, 4);
        int tiles_y = DivideRoundUp(dst_shape.h, 4);
        const BHWC shape_0{src_shape.b, 36, tiles_x * tiles_y, src_shape.c};
        const BHWC shape_1{src_shape.b, 36, tiles_x * tiles_y, dst_shape.c};

        Winograd4x4To36Attributes wino_up_attr;
        wino_up_attr.padding = attr.padding;
        (*last_node_id) += 1;
        int value_id = *last_value_id + 1;
        (*tensor_shapes)[value_id] = shape_0;
        (*tensor_shapes)[value_id + 1] = shape_1;
        *tasks = SelectWinograd4x4To36(inputs[0], value_id, wino_up_attr,
                                       gpu_info, options);

        (*last_node_id) += 1;
        auto gpu_op = ConvolutionWino4x4To6x6(value_id, value_id + 1, shape_1,
                                              attr, gpu_info, options);
        tasks->push_back(
            std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op)));

        Winograd36To4x4Attributes wino_down_attr;
        wino_down_attr.output_shape = dst_shape;
        wino_down_attr.biases = attr.bias;
        (*last_node_id) += 1;
        auto t2 = SelectWinograd36To4x4(value_id + 1, outputs[0],
                                        wino_down_attr, gpu_info, options);
        tasks->insert(tasks->end(), t2.begin(), t2.end());
        (*last_value_id) += 2;
      } else {
        auto gpu_op = ConvolutionGeneric(inputs[0], outputs[0], dst_shape, attr,
                                         gpu_info, options);
        *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      }
      break;
    }
    case OperationType::CONVOLUTION_TRANSPOSED:
      if (graph.FindInputs(node->id).size() != 1) {
        return absl::UnimplementedError(
            "Convolution Transposed does not support more than 1 runtime "
            "tensor");
      }
      *tasks = SelectConvolutionTransposed(
          inputs[0], outputs[0],
          absl::any_cast<ConvolutionTransposedAttributes>(
              node->operation.attributes),
          gpu_info, options);
      break;
    case OperationType::DEPTHWISE_CONVOLUTION:
      if (graph.FindInputs(node->id).size() != 1) {
        return absl::UnimplementedError(
            "DepthWise Convolution does not support more than 1 runtime "
            "tensor");
      }
      *tasks =
          SelectDepthWiseConv(inputs[0], outputs[0],
                              absl::any_cast<DepthwiseConvolution2DAttributes>(
                                  node->operation.attributes),
                              options);
      break;
    case OperationType::FULLY_CONNECTED: {
      auto gpu_op = FullyConnected(
          inputs[0], outputs[0],
          absl::any_cast<FullyConnectedAttributes>(node->operation.attributes),
          gpu_info, options);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::MAX_UNPOOLING_2D: {
      auto gpu_op = MaxUnpooling(
          inputs[0], inputs[1], outputs[0],
          absl::any_cast<MaxUnpooling2DAttributes>(node->operation.attributes));
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::MEAN: {
      auto attr = absl::any_cast<MeanAttributes>(node->operation.attributes);
      if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
        return absl::UnimplementedError("Mean supports HW axis only in Metal");
      }
      auto gpu_op = Mean(inputs[0], outputs[0], attr);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::MUL:
      if (inputs.size() == 1) {
        if (node->operation.attributes.has_value()) {
          auto attr =
              absl::any_cast<ElementwiseAttributes>(node->operation.attributes);
          auto gpu_op = ElementwiseWithOneInputAndConstantArguent(
              inputs[0], outputs[0], options, op_type, attr.param);
          *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
        } else {
          return absl::UnimplementedError(
              "Missing attributes for single input op: " +
              node->operation.type);
        }
      } else if (inputs.size() == 2) {
        const auto srcs = graph.FindInputs(node_id);
        auto gpu_op = ElementwiseWithTwoInputs(inputs, outputs[0],
                                               srcs[1]->tensor.shape, op_type);
        *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      }
      break;
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node->operation.attributes);
      if (attr.appended.b != 0 || attr.prepended.b != 0) {
        return absl::UnimplementedError("Padding for BATCH is not supported.");
      }
      auto gpu_op = Padding(inputs[0], outputs[0], attr);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::POOLING_2D: {
      auto attr =
          absl::any_cast<Pooling2DAttributes>(node->operation.attributes);
      auto gpu_op = Pooling(inputs[0], outputs[0], attr, false);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      if (attr.type == PoolingType::MAX && attr.output_indices) {
        auto gpu_ind_op = Pooling(inputs[0], outputs[1], attr, true);
        tasks->push_back(
            std::make_shared<ComputeTaskDescriptor>(std::move(gpu_ind_op)));
      }
      break;
    }
    case OperationType::PRELU: {
      const auto src_shape = graph.FindInputs(node_id)[0]->tensor.shape;
      *tasks = SelectPReLU(
          src_shape, inputs[0], outputs[0],
          absl::any_cast<PReLUAttributes>(node->operation.attributes), options);
      break;
    }
    case OperationType::RELU: {
      auto gpu_op =
          ReLU(inputs[0], outputs[0],
               absl::any_cast<ReLUAttributes>(node->operation.attributes));
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::QUANTIZE_AND_DEQUANTIZE:
      *tasks = SelectQuantizeAndDequantize(
          inputs[0], outputs[0],
          absl::any_cast<QuantizeAndDequantizeAttributes>(
              node->operation.attributes));
      break;
    case OperationType::RESHAPE: {
      const auto src_shape = graph.FindInputs(node_id)[0]->tensor.shape;
      *tasks = SelectReshape(
          src_shape, inputs[0], outputs[0],
          absl::any_cast<ReshapeAttributes>(node->operation.attributes));
      break;
    }
    case OperationType::RESIZE: {
      auto gpu_op = Resize(
          inputs[0], outputs[0],
          absl::any_cast<Resize2DAttributes>(node->operation.attributes));
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::SLICE: {
      auto gpu_op =
          Slice(inputs[0], outputs[0],
                absl::any_cast<SliceAttributes>(node->operation.attributes));
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::SOFTMAX: {
      auto attr = absl::any_cast<SoftmaxAttributes>(node->operation.attributes);
      if (attr.axis != Axis::CHANNELS) {
        return absl::UnimplementedError(
            "Softmax supports only CHANNELS dimension");
      }
      const auto src_shape = graph.FindInputs(node_id)[0]->tensor.shape;
      *tasks = SelectSoftmax(src_shape, inputs[0], outputs[0], gpu_info);
      break;
    }
    case OperationType::SPACE_TO_DEPTH:
      *tasks = SelectSpaceToDepth(
          inputs[0], outputs[0],
          absl::any_cast<SpaceToDepthAttributes>(node->operation.attributes));
      break;
    case OperationType::ABS:
    case OperationType::COPY:
    case OperationType::COS:
    case OperationType::ELU:
    case OperationType::EXP:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::NEG:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH: {
      auto gpu_op = ElementwiseWithOneInput(inputs[0], outputs[0], op_type);
      *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      break;
    }
    case OperationType::DIV:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      if (inputs.size() == 1) {
        if (node->operation.attributes.has_value()) {
          auto attr =
              absl::any_cast<ElementwiseAttributes>(node->operation.attributes);
          auto gpu_op = ElementwiseWithOneInputAndConstantArguent(
              inputs[0], outputs[0], options, op_type, attr.param);
          *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
        } else {
          return absl::UnimplementedError(
              "Missing attributes for single input op: " +
              node->operation.type);
        }
      } else if (inputs.size() == 2) {
        const auto srcs = graph.FindInputs(node_id);
        auto gpu_op = ElementwiseWithTwoInputs(inputs, outputs[0],
                                               srcs[1]->tensor.shape, op_type);
        *tasks = {std::make_shared<ComputeTaskDescriptor>(std::move(gpu_op))};
      }
    } break;
    case OperationType::BATCH_NORMALIZATION:
    case OperationType::BATCH_TO_SPACE:
    case OperationType::BATCHED_MATMUL:
    case OperationType::CONST:
    case OperationType::LSTM:
    // TODO(b/162763635): implement MeanStddevNormalization for Metal.
    case OperationType::MEAN_STDDEV_NORMALIZATION:
    case OperationType::REDUCE_MAXIMUM:
    case OperationType::REDUCE_MINIMUM:
    case OperationType::REDUCE_PRODUCT:
    case OperationType::REDUCE_SUM:
    // comparison operations
    case OperationType::LESS:
    case OperationType::LESS_EQUAL:
    case OperationType::EQUAL:
    case OperationType::NOT_EQUAL:
    case OperationType::GREATER:
    case OperationType::GREATER_EQUAL:
    case OperationType::SPACE_TO_BATCH:
    case OperationType::TRANSPOSE:
    case OperationType::UNKNOWN:
      return absl::UnimplementedError("Unsupported op: " +
                                      node->operation.type);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Compile(const GraphFloat32& graph, const GpuInfo& gpu_info,
                     const RuntimeOptions& options,
                     CompiledModel* compiled_model) {
  int last_node_id = 0;
  for (const auto& node : graph.nodes()) {
    last_node_id = std::max(last_node_id, static_cast<int>(node->id));
  }
  int last_value_id = 0;
  for (const auto& value : graph.values()) {
    compiled_model->tensor_shapes[value->id] = value->tensor.shape;
    last_value_id = std::max(last_value_id, static_cast<int>(value->id));
  }
  int node_linear_id = 0;
  for (const auto& node : graph.nodes()) {
    std::vector<ValueId> inputs;
    for (auto& input : graph.FindInputs(node->id)) {
      inputs.push_back(static_cast<ValueId>(input->id));
    }
    std::vector<ValueId> outputs;
    for (auto& output : graph.FindOutputs(node->id)) {
      outputs.push_back(static_cast<ValueId>(output->id));
    }
    std::vector<ComputeTaskDescriptorPtr> tasks;
    auto custom_status =
        RegisterCustomOps(graph, node, inputs, outputs, options, &tasks);
    if (!custom_status.ok()) {
      auto primary_status = RegisterPrimaryOps(
          graph, node, inputs, outputs, gpu_info, options, &last_node_id,
          &last_value_id, &compiled_model->tensor_shapes, &tasks);
      if (!primary_status.ok()) {
        return absl::UnimplementedError(
            absl::Substitute("Unsupported op type: $0; custom registry error: "
                             "$1; primary registry error: $2;",
                             node->operation.type, custom_status.message(),
                             primary_status.message()));
      }
    }
    for (auto& task : tasks) {
      NodeDescriptor node_desc;
      node_desc.task = task;
      node_desc.description =
          node->operation.type + "_" + std::to_string(node->id);
      node_desc.id = node_linear_id++;
      compiled_model->nodes.push_back(node_desc);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
