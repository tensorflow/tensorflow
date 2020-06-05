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
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/environment.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/concat.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/custom_registry.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/mul.h"
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

bool IsWidthBroadcastedForSecondInput(const std::vector<Value*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.w != inputs[1]->tensor.shape.w &&
         inputs[1]->tensor.shape.w == 1;
}
bool IsHeightBroadcastedForSecondInput(const std::vector<Value*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.h != inputs[1]->tensor.shape.h &&
         inputs[1]->tensor.shape.h == 1;
}
bool IsChannelsBroadcastedForSecondInput(const std::vector<Value*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.c != inputs[1]->tensor.shape.c &&
         inputs[1]->tensor.shape.c == 1;
}

std::vector<ComputeTaskDescriptorPtr> SelectDepthWiseConv(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const metal::RuntimeOptions& options) {
  if (CheckDepthWiseConv3x3Stride1x1Support(attr)) {
    return DepthWiseConv3x3Stride1x1(id, input_id, output_id, attr, options);
  } else if (CheckDepthWiseConv3x3Stride2Support(attr)) {
    return DepthWiseConv3x3Stride2(id, input_id, output_id, attr, options);
  } else {
    return DepthWiseConvolution(id, input_id, output_id, attr, options);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectConvolutionTransposed(
    int id, ValueId input_id, ValueId output_id,
    const ConvolutionTransposedAttributes& attr, const DeviceInfo& device_info,
    const metal::RuntimeOptions& options) {
  if (CheckConvolutionTransposed4x4Support(attr)) {
    return ConvolutionTransposed4x4(id, input_id, output_id, attr, device_info,
                                    options);
  } else {
    return ConvolutionTransposed(id, input_id, output_id, attr, device_info,
                                 options);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectQuantizeAndDequantize(
    int id, ValueId input_id, ValueId output_id,
    const QuantizeAndDequantizeAttributes& attr) {
  return QuantizeAndDequantize(id, input_id, output_id, attr);
}

std::vector<ComputeTaskDescriptorPtr> SelectPReLU(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const PReLUAttributes& attr, const metal::RuntimeOptions& options) {
  auto alpha = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha) {
    return PReLU(id, input_id, output_id, attr, options);
  }
  auto alpha3d = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha3d) {
    return {};
  }
  const auto shape = graph.FindInputs(id)[0]->tensor.shape;
  if (alpha3d->shape.h != shape.h || alpha3d->shape.w != shape.w ||
      alpha3d->shape.c != shape.c) {
    return {};
  }
  return PReLUFull(id, input_id, output_id, attr, options);
}

std::vector<ComputeTaskDescriptorPtr> SelectReshape(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const ReshapeAttributes& attr) {
  const auto src_shape = graph.FindInputs(id)[0]->tensor.shape;
  if (src_shape.c % 4 == 0 && attr.new_shape.c % 4 == 0) {
    return Reshapex4(id, input_id, output_id, attr);
  } else {
    return Reshape(id, input_id, output_id, attr);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectSoftmax(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const DeviceInfo& device_info) {
  const auto src_shape = graph.FindInputs(id)[0]->tensor.shape;
  if (src_shape.w == 1 && src_shape.h == 1) {
    return Softmax1x1(id, input_id, output_id, device_info, src_shape.c);
  } else {
    return Softmax(id, input_id, output_id, src_shape.c);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectSpaceToDepth(
    const GraphFloat32& graph, int id, ValueId input_id, ValueId output_id,
    const SpaceToDepthAttributes& attr) {
  return SpaceToDepth(id, input_id, output_id, attr);
}

std::vector<ComputeTaskDescriptorPtr> SelectWinograd4x4To36(
    int id, ValueId input_id, ValueId output_id,
    const Winograd4x4To36Attributes& attr, const DeviceInfo& device_info,
    const metal::RuntimeOptions& options) {
  if (device_info.IsAppleGPU()) {
    return Winograd4x4To36(id, input_id, output_id, attr);
  } else {
    return Winograd4x4To36TileX6(id, input_id, output_id, attr, options);
  }
}

std::vector<ComputeTaskDescriptorPtr> SelectWinograd36To4x4(
    int id, ValueId input_id, ValueId output_id,
    const Winograd36To4x4Attributes& attr, const DeviceInfo& device_info,
    const metal::RuntimeOptions& options) {
  if (device_info.IsAppleGPU()) {
    return Winograd36To4x4(id, input_id, output_id, options, attr);
  } else {
    return Winograd36To4x4Tile4x1(id, input_id, output_id, options, attr);
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
                                const DeviceInfo& device_info,
                                const RuntimeOptions& options,
                                int* last_node_id, int* last_value_id,
                                std::vector<ComputeTaskDescriptorPtr>* tasks) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return absl::InvalidArgumentError(
        "Only identical batch dimension is supported");
  }
  int node_id = static_cast<int>(node->id);
  auto op_type = OperationTypeFromString(node->operation.type);
  switch (op_type) {
    case OperationType::ADD: {
      const auto srcs = graph.FindInputs(node_id);
      ElementwiseBroadcastSettings broadcast;
      broadcast.width = IsWidthBroadcastedForSecondInput(srcs);
      broadcast.height = IsHeightBroadcastedForSecondInput(srcs);
      broadcast.channels = IsChannelsBroadcastedForSecondInput(srcs);
      if (broadcast.width || broadcast.height || broadcast.channels) {
        *tasks = ElementwiseWithTwoInputs(node_id, inputs, outputs[0], op_type,
                                          broadcast);
      } else {
        const AddAttributes& attr =
            absl::any_cast<AddAttributes>(node->operation.attributes);
        const auto* hwc_tensor =
            absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(
                &attr.param);
        if (hwc_tensor) {
          return absl::UnimplementedError(
              "Unsupported op: " + node->operation.type +
              ", no support of HWC constant tensor");
        }
        *tasks = Add(node_id, inputs, outputs[0], attr, options);
      }
      break;
    }
    case OperationType::CONCAT: {
      std::vector<BHWC> input_shapes;
      for (auto& input : graph.FindInputs(node->id)) {
        input_shapes.push_back(input->tensor.shape);
      }
      *tasks =
          Concat(node_id, inputs, outputs[0],
                 absl::any_cast<ConcatAttributes>(node->operation.attributes),
                 input_shapes);
      break;
    }
    case OperationType::CONVOLUTION_2D: {
      if (graph.FindInputs(node->id).size() != 1) {
        return absl::UnimplementedError(
            "Convolution does not support more than 1 runtime tensor");
      }
      const auto dst_shape = graph.FindOutputs(node_id)[0]->tensor.shape;
      auto attr =
          absl::any_cast<Convolution2DAttributes>(node->operation.attributes);
      if (IsSuitableForWinograd4x4To6x6(attr, dst_shape)) {
        int tiles_x = DivideRoundUp(dst_shape.w, 4);
        int tiles_y = DivideRoundUp(dst_shape.h, 4);

        Winograd4x4To36Attributes wino_up_attr;
        wino_up_attr.padding = attr.padding;
        (*last_node_id) += 1;
        int value_id = *last_value_id + 1;
        *tasks = SelectWinograd4x4To36(*last_node_id, inputs[0], value_id,
                                       wino_up_attr, device_info, options);

        BHWC conv_shape{dst_shape.b, 36, tiles_x * tiles_y, dst_shape.c};
        (*last_node_id) += 1;
        auto t1 =
            ConvolutionWino4x4To6x6(*last_node_id, value_id, value_id + 1,
                                    conv_shape, attr, device_info, options);
        tasks->insert(tasks->end(), t1.begin(), t1.end());

        Winograd36To4x4Attributes wino_down_attr;
        wino_down_attr.output_shape = dst_shape;
        wino_down_attr.biases = attr.bias;
        (*last_node_id) += 1;
        auto t2 = SelectWinograd36To4x4(*last_node_id, value_id + 1, outputs[0],
                                        wino_down_attr, device_info, options);
        tasks->insert(tasks->end(), t2.begin(), t2.end());
        (*last_value_id) += 2;
      } else {
        *tasks = ConvolutionGeneric(node_id, inputs[0], outputs[0], dst_shape,
                                    attr, device_info, options);
      }
      break;
    }
    case OperationType::CONVOLUTION_TRANSPOSED:
      *tasks = SelectConvolutionTransposed(
          node_id, inputs[0], outputs[0],
          absl::any_cast<ConvolutionTransposedAttributes>(
              node->operation.attributes),
          device_info, options);
      break;
    case OperationType::DEPTHWISE_CONVOLUTION:
      *tasks =
          SelectDepthWiseConv(node_id, inputs[0], outputs[0],
                              absl::any_cast<DepthwiseConvolution2DAttributes>(
                                  node->operation.attributes),
                              options);
      break;
    case OperationType::FULLY_CONNECTED:
      *tasks = FullyConnected(
          node_id, inputs[0], outputs[0],
          absl::any_cast<FullyConnectedAttributes>(node->operation.attributes),
          device_info, options);
      break;
    case OperationType::MAX_UNPOOLING_2D:
      *tasks = MaxUnpooling(
          node_id, inputs[0], inputs[1], outputs[0],
          absl::any_cast<MaxUnpooling2DAttributes>(node->operation.attributes));
      break;
    case OperationType::MEAN:
      *tasks = Mean(node_id, inputs[0], outputs[0],
                    absl::any_cast<MeanAttributes>(node->operation.attributes));
      break;
    case OperationType::MUL:
      if (node->operation.attributes.has_value()) {
        const MultiplyAttributes& attr =
            absl::any_cast<MultiplyAttributes>(node->operation.attributes);
        const auto* hwc_tensor =
            absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(
                &attr.param);
        if (hwc_tensor) {
          return absl::UnimplementedError(
              "Unsupported op: " + node->operation.type +
              ", no support of HWC constant tensor");
        }
        *tasks = Multiply(node_id, inputs[0], outputs[0], attr, options);
      } else {
        if (inputs.size() == 2) {
          const auto srcs = graph.FindInputs(node_id);
          ElementwiseBroadcastSettings broadcast;
          broadcast.width = IsWidthBroadcastedForSecondInput(srcs);
          broadcast.height = IsHeightBroadcastedForSecondInput(srcs);
          broadcast.channels = IsChannelsBroadcastedForSecondInput(srcs);
          *tasks = ElementwiseWithTwoInputs(node_id, inputs, outputs[0],
                                            op_type, broadcast);
        } else {
          return absl::UnimplementedError(
              "No support of multiply with more than 2 inputs");
        }
      }
      break;
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node->operation.attributes);
      if (attr.appended.b != 0 || attr.prepended.b != 0) {
        return absl::UnimplementedError("Padding for BATCH is not supported.");
      }
      *tasks = Padding(node_id, inputs[0], outputs[0], attr);
      break;
    }
    case OperationType::POOLING_2D:
      *tasks = Pooling(
          node_id, inputs[0], outputs,
          absl::any_cast<Pooling2DAttributes>(node->operation.attributes));
      break;
    case OperationType::PRELU:
      *tasks = SelectPReLU(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<PReLUAttributes>(node->operation.attributes), options);
      break;
    case OperationType::RELU:
      *tasks = ReLU(node_id, inputs[0], outputs[0],
                    absl::any_cast<ReLUAttributes>(node->operation.attributes));
      break;
    case OperationType::QUANTIZE_AND_DEQUANTIZE:
      *tasks = SelectQuantizeAndDequantize(
          node_id, inputs[0], outputs[0],
          absl::any_cast<QuantizeAndDequantizeAttributes>(
              node->operation.attributes));
      break;
    case OperationType::RESHAPE:
      *tasks = SelectReshape(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<ReshapeAttributes>(node->operation.attributes));
      break;
    case OperationType::RESIZE:
      *tasks = Resize(
          node_id, inputs[0], outputs[0],
          absl::any_cast<Resize2DAttributes>(node->operation.attributes));
      break;
    case OperationType::SLICE:
      *tasks =
          Slice(node_id, inputs[0], outputs[0],
                absl::any_cast<SliceAttributes>(node->operation.attributes));
      break;
    case OperationType::SOFTMAX: {
      auto attr = absl::any_cast<SoftmaxAttributes>(node->operation.attributes);
      if (attr.axis != Axis::CHANNELS) {
        return absl::UnimplementedError(
            "Softmax supports only CHANNELS dimension");
      }
      *tasks =
          SelectSoftmax(graph, node_id, inputs[0], outputs[0], device_info);
      break;
    }
    case OperationType::SPACE_TO_DEPTH:
      *tasks = SelectSpaceToDepth(
          graph, node_id, inputs[0], outputs[0],
          absl::any_cast<SpaceToDepthAttributes>(node->operation.attributes));
      break;
    case OperationType::ABS:
    case OperationType::COS:
    case OperationType::EXP:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH:
      *tasks = ElementwiseWithOneInput(node_id, inputs[0], outputs[0], op_type);
      break;
    case OperationType::DIV:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      const ElementwiseAttributes* attr =
          absl::any_cast<ElementwiseAttributes>(&node->operation.attributes);
      if (attr) {
        const auto* hwc_tensor =
            absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(
                &attr->param);
        if (hwc_tensor) {
          return absl::UnimplementedError(
              "Unsupported op: " + node->operation.type +
              ", no support of HWC constant tensor");
        }
        *tasks = ElementwiseWithOneInputAndConstantArguent(
            node_id, inputs[0], outputs[0], options, op_type, *attr);
      } else {
        const auto srcs = graph.FindInputs(node_id);
        ElementwiseBroadcastSettings broadcast;
        broadcast.width = IsWidthBroadcastedForSecondInput(srcs);
        broadcast.height = IsHeightBroadcastedForSecondInput(srcs);
        broadcast.channels = IsChannelsBroadcastedForSecondInput(srcs);
        *tasks = ElementwiseWithTwoInputs(node_id, inputs, outputs[0], op_type,
                                          broadcast);
      }
    } break;
    case OperationType::BATCH_NORMALIZATION:
    case OperationType::BATCH_TO_SPACE:
    case OperationType::CONST:
    case OperationType::LSTM:
    case OperationType::SPACE_TO_BATCH:
    case OperationType::TRANSPOSE:
    case OperationType::UNKNOWN:
      return absl::UnimplementedError("Unsupported op: " +
                                      node->operation.type);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Compile(const GraphFloat32& graph, const DeviceInfo& device_info,
                     const RuntimeOptions& options,
                     CompiledModel* compiled_model) {
  int last_node_id = 0;
  for (const auto& node : graph.nodes()) {
    last_node_id = std::max(last_node_id, static_cast<int>(node->id));
  }
  int last_value_id = 0;
  for (const auto& value : graph.values()) {
    last_value_id = std::max(last_value_id, static_cast<int>(value->id));
  }
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
      auto primary_status =
          RegisterPrimaryOps(graph, node, inputs, outputs, device_info, options,
                             &last_node_id, &last_value_id, &tasks);
      if (!primary_status.ok()) {
        return absl::UnimplementedError(
            absl::Substitute("Unsupported op type: $0; custom registry error: "
                             "$1; primary registry error: $2;",
                             node->operation.type, custom_status.message(),
                             primary_status.message()));
      }
    }
    for (const auto& task : tasks) {
      task->description = node->operation.type + "_" + std::to_string(node->id);
    }
    compiled_model->insert(compiled_model->end(), tasks.begin(), tasks.end());
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
