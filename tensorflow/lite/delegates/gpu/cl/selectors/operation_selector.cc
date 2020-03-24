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

#include "tensorflow/lite/delegates/gpu/cl/selectors/operation_selector.h"

#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_transposed_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/default_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/dw_convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/fully_connected_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/simple_selectors.h"
#include "tensorflow/lite/delegates/gpu/cl/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
bool IsWidthBroadcastedForSecondInput(
    const std::vector<Value<TensorRef<BHWC>>*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.w != inputs[1]->tensor.shape.w &&
         inputs[1]->tensor.shape.w == 1;
}
bool IsHeightBroadcastedForSecondInput(
    const std::vector<Value<TensorRef<BHWC>>*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.h != inputs[1]->tensor.shape.h &&
         inputs[1]->tensor.shape.h == 1;
}
bool IsChannelsBroadcastedForSecondInput(
    const std::vector<Value<TensorRef<BHWC>>*>& inputs) {
  return inputs.size() == 2 &&
         inputs[0]->tensor.shape.c != inputs[1]->tensor.shape.c &&
         inputs[1]->tensor.shape.c == 1;
}

bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr,
                                   const CLDevice& device,
                                   const BHWC& dst_shape) {
  const int tiles_x = IntegralDivideRoundUp(dst_shape.w, 4);
  const int tiles_y = IntegralDivideRoundUp(dst_shape.h, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const bool suitable_attributes =
      attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
      attr.dilations == HW(1, 1) && attr.strides == HW(1, 1);
  // Mali among other devices has smaller SIMD line size
  const int min_depth = device.IsMali() ? 16 : 32;
  const int min_hw = device.IsMali() ? 32 : 128;
  const bool recommended_channels =
      dst_depth % 4 == 0 && src_depth >= min_depth && dst_depth >= min_depth;
  const bool recommended_hw = tiles_x * tiles_y >= min_hw;
  return suitable_attributes && recommended_channels && recommended_hw;
}

Status WinogradFromNode(const CreationContext& creation_context,
                        const OperationDef& op_def, ModelHints hints,
                        const BHWC& input_shape, const BHWC& output_shape,
                        const Convolution2DAttributes& attr,
                        GPUOperationsSubgraph* gpu_subgraph) {
  if (!IsSuitableForWinograd4x4To6x6(attr, *creation_context.device,
                                     output_shape)) {
    return UnimplementedError("No implementation for this case.");
  }

  const int tiles_x = IntegralDivideRoundUp(output_shape.w, 4);
  const int tiles_y = IntegralDivideRoundUp(output_shape.h, 4);
  const BHWC shape_0{input_shape.b, 36, tiles_x * tiles_y, input_shape.c};
  const BHWC shape_1{input_shape.b, 36, tiles_x * tiles_y, output_shape.c};
  TensorDescriptor td_0;
  td_0.storage_type = SelectBestStorageType(
      *creation_context.context, *creation_context.device, shape_0,
      op_def.src_tensors[0].storage_type, op_def.src_tensors[0].data_type,
      op_def.src_tensors[0].layout);
  td_0.data_type = op_def.src_tensors[0].data_type;
  td_0.layout = op_def.src_tensors[0].layout;
  TensorDescriptor td_1;
  td_1.storage_type = SelectBestStorageType(
      *creation_context.context, *creation_context.device, shape_1,
      op_def.src_tensors[0].storage_type, op_def.src_tensors[0].data_type,
      op_def.src_tensors[0].layout);
  td_1.data_type = op_def.src_tensors[0].data_type;
  td_1.layout = op_def.src_tensors[0].layout;
  gpu_subgraph->new_tensors = {{shape_0, td_0}, {shape_1, td_1}};
  gpu_subgraph->operations.clear();
  gpu_subgraph->operations.resize(3);

  OperationDef winograd_up_def;
  winograd_up_def.precision = op_def.precision;
  winograd_up_def.src_tensors.push_back(op_def.src_tensors[0]);
  winograd_up_def.dst_tensors.push_back(td_0);
  auto& winograd_up = gpu_subgraph->operations[0];
  RETURN_IF_ERROR(SelectWinograd4x4To36(
      creation_context, attr.padding, winograd_up_def, &winograd_up.operation));
  winograd_up.input_ids = {0};
  winograd_up.output_ids = {-1};

  OperationDef conv_def;
  conv_def.precision = op_def.precision;
  conv_def.src_tensors.push_back(td_0);
  conv_def.dst_tensors.push_back(td_1);
  auto& conv = gpu_subgraph->operations[1];
  conv.input_ids = {-1};
  conv.output_ids = {-2};
  RETURN_IF_ERROR(SelectConvolutionForWinograd(
      attr, input_shape, creation_context, conv_def, hints, &conv.operation));

  OperationDef winograd_down_def;
  winograd_down_def.precision = op_def.precision;
  winograd_down_def.src_tensors.push_back(td_1);
  winograd_down_def.dst_tensors.push_back(op_def.dst_tensors[0]);
  auto& winograd_down = gpu_subgraph->operations[2];
  winograd_down.input_ids = {-2};
  winograd_down.output_ids = {0};
  auto bias_copy = attr.bias;
  if (bias_copy.shape.v < attr.weights.shape.o) {
    bias_copy.shape = Linear(attr.weights.shape.o);
    bias_copy.data.resize(attr.weights.shape.o);
  }
  RETURN_IF_ERROR(SelectWinograd36To4x4(creation_context, winograd_down_def,
                                        bias_copy, &winograd_down.operation));

  return OkStatus();
}

}  // namespace

Status GPUOperationFromNode(const CreationContext& creation_context,
                            const OperationDef& op_def, ModelHints hints,
                            const std::vector<Value<TensorRef<BHWC>>*>& inputs,
                            const std::vector<Value<TensorRef<BHWC>>*>& outputs,
                            const Node& node,
                            GPUOperationsSubgraph* gpu_subgraph) {
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
  auto op_type = OperationTypeFromString(node.operation.type);
  switch (op_type) {
    case OperationType::ADD: {
      const auto attr =
          absl::any_cast<AddAttributes>(node.operation.attributes);
      const auto* adds =
          absl::get_if<::tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
              &attr.param);
      const auto* adds_scalar = absl::get_if<float>(&attr.param);
      if (adds || adds_scalar) {
        return SelectBroadcastAdd(attr, creation_context, op_def, gpu_op);
      } else {
        BroadcastSettings broadcast;
        broadcast.width = IsWidthBroadcastedForSecondInput(inputs);
        broadcast.height = IsHeightBroadcastedForSecondInput(inputs);
        broadcast.channels = IsChannelsBroadcastedForSecondInput(inputs);
        if (broadcast.width || broadcast.height || broadcast.channels) {
          ElementwiseTwoInput operation =
              CreateElementwiseTwoInput(op_def, op_type, broadcast);
          *gpu_op =
              absl::make_unique<ElementwiseTwoInput>(std::move(operation));
        } else {
          auto output = outputs[0];
          std::vector<int> channels(inputs.size());
          for (int i = 0; i < inputs.size(); ++i) {
            channels[i] = inputs[i]->tensor.shape.c;
          }
          SelectAdd(op_def, channels, output->tensor.shape.c, gpu_op);
        }
        return OkStatus();
      }
    }
    case OperationType::CONCAT: {
      auto attr = absl::any_cast<ConcatAttributes>(node.operation.attributes);
      std::vector<int> channels(inputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        channels[i] = inputs[i]->tensor.shape.c;
      }
      return SelectConcat(attr, channels, op_def, gpu_op);
    }
    case OperationType::CONVOLUTION_2D: {
      auto attr =
          absl::any_cast<Convolution2DAttributes>(node.operation.attributes);
      auto input_shape = inputs[0]->tensor.shape;
      auto output_shape = outputs[0]->tensor.shape;
      if (WinogradFromNode(creation_context, op_def, hints, input_shape,
                           output_shape, attr, gpu_subgraph)
              .ok()) {
        return OkStatus();
      } else {
        gpu_op = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
        return SelectConvolution(attr, output_shape, creation_context, op_def,
                                 hints, gpu_op);
      }
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      auto attr = absl::any_cast<ConvolutionTransposedAttributes>(
          node.operation.attributes);
      return SelectConvolutionTransposed(attr, creation_context, op_def,
                                         gpu_op);
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      auto attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
          node.operation.attributes);
      return SelectDWConvolution(attr, creation_context, op_def, gpu_op);
    }
    case OperationType::FULLY_CONNECTED: {
      auto attr =
          absl::any_cast<FullyConnectedAttributes>(node.operation.attributes);
      return SelectFullyConnected(attr, creation_context, op_def,
                                  inputs[0]->tensor.shape.b, gpu_op);
    }
    case OperationType::LSTM: {
      SelectLSTM(op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      auto attr =
          absl::any_cast<MaxUnpooling2DAttributes>(node.operation.attributes);
      SelectMaxUnpooling(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::MEAN: {
      auto attr = absl::any_cast<MeanAttributes>(node.operation.attributes);
      return SelectMean(attr, op_def, gpu_op);
    }
    case OperationType::MUL: {
      if (node.operation.attributes.has_value()) {
        auto attr =
            absl::any_cast<MultiplyAttributes>(node.operation.attributes);

        return SelectMultiplyScalar(attr, creation_context, op_def, gpu_op);
      } else {
        if (inputs.size() == 2) {
          BroadcastSettings broadcast;
          broadcast.width = IsWidthBroadcastedForSecondInput(inputs);
          broadcast.height = IsHeightBroadcastedForSecondInput(inputs);
          broadcast.channels = IsChannelsBroadcastedForSecondInput(inputs);
          ElementwiseTwoInput operation =
              CreateElementwiseTwoInput(op_def, op_type, broadcast);
          *gpu_op =
              absl::make_unique<ElementwiseTwoInput>(std::move(operation));
          return OkStatus();
        } else {
          return UnimplementedError(
              "No support of multiply with more than 2 inputs");
        }
        return OkStatus();
      }
    }
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
      SelectPadding(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::POOLING_2D: {
      auto attr =
          absl::any_cast<Pooling2DAttributes>(node.operation.attributes);
      SelectPooling(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::PRELU: {
      auto attr = absl::any_cast<PReLUAttributes>(node.operation.attributes);
      return SelectPReLU(attr, creation_context, op_def, gpu_op);
    }
    case OperationType::QUANTIZE_AND_DEQUANTIZE: {
      auto attr = absl::any_cast<QuantizeAndDequantizeAttributes>(
          node.operation.attributes);
      return SelectQuantizeAndDequantize(attr, creation_context, op_def,
                                         gpu_op);
    }
    case OperationType::RELU: {
      auto attr = absl::any_cast<ReLUAttributes>(node.operation.attributes);
      SelectReLU(creation_context, attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::RESHAPE: {
      const int src_channels = inputs[0]->tensor.shape.c;
      auto attr = absl::any_cast<ReshapeAttributes>(node.operation.attributes);
      SelectReshape(src_channels, attr.new_shape.c, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::RESIZE: {
      auto attr = absl::any_cast<Resize2DAttributes>(node.operation.attributes);
      return SelectResize(attr, op_def, gpu_op);
    }
    case OperationType::SLICE: {
      auto attr = absl::any_cast<SliceAttributes>(node.operation.attributes);
      SelectStridedSlice(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::SOFTMAX: {
      SelectSoftmax(inputs[0]->tensor.shape, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::SPACE_TO_DEPTH: {
      auto attr =
          absl::any_cast<SpaceToDepthAttributes>(node.operation.attributes);
      SelectSpaceToDepth(attr, op_def, gpu_op);
      return OkStatus();
    }
    case OperationType::TRANSPOSE: {
      auto attr =
          absl::any_cast<TransposeAttributes>(node.operation.attributes);
      SelectTranspose(attr, op_def, gpu_op);
      return OkStatus();
    }
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
    case OperationType::TANH: {
      ElementwiseOneInput operation =
          CreateElementwiseOneInput(op_def, op_type);
      *gpu_op = absl::make_unique<ElementwiseOneInput>(std::move(operation));
      return OkStatus();
    }
    case OperationType::DIV:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      BroadcastSettings broadcast;
      broadcast.width = IsWidthBroadcastedForSecondInput(inputs);
      broadcast.height = IsHeightBroadcastedForSecondInput(inputs);
      broadcast.channels = IsChannelsBroadcastedForSecondInput(inputs);
      const ElementwiseAttributes* attr =
          absl::any_cast<ElementwiseAttributes>(&node.operation.attributes);
      ElementwiseTwoInput operation = CreateElementwiseTwoInput(
          creation_context, op_def, op_type, broadcast, attr);
      *gpu_op = absl::make_unique<ElementwiseTwoInput>(std::move(operation));
      return OkStatus();
    }
    default:
      return SelectDefault(creation_context, op_def, hints, inputs, outputs,
                           node, gpu_subgraph);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
