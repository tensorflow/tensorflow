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
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr,
                                   const CLDevice& device,
                                   const BHWC& dst_shape) {
  const int tiles_x = DivideRoundUp(dst_shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_shape.h, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
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

absl::Status WinogradFromNode(const CreationContext& creation_context,
                              const OperationDef& op_def, ModelHints hints,
                              const BHWC& input_shape, const BHWC& output_shape,
                              const Convolution2DAttributes& attr,
                              GPUOperationsSubgraph* gpu_subgraph) {
  if (!IsSuitableForWinograd4x4To6x6(attr, *creation_context.device,
                                     output_shape)) {
    return absl::UnimplementedError("No implementation for this case.");
  }

  const int tiles_x = DivideRoundUp(output_shape.w, 4);
  const int tiles_y = DivideRoundUp(output_shape.h, 4);
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
  return absl::OkStatus();
}

}  // namespace

absl::Status GPUOperationFromNode(const CreationContext& creation_context,
                                  const OperationDef& op_def, ModelHints hints,
                                  const std::vector<Value*>& inputs,
                                  const std::vector<Value*>& outputs,
                                  const Node& node,
                                  GPUOperationsSubgraph* gpu_subgraph) {
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
  auto op_type = OperationTypeFromString(node.operation.type);
  switch (op_type) {
    case OperationType::ADD: {
      auto attr = absl::any_cast<AddAttributes>(node.operation.attributes);
      const float* scalar = absl::get_if<float>(&attr.param);
      const auto* linear_tensor =
          absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
              &attr.param);
      if (scalar) {
        ElementwiseOneRuntimeOneScalar operation =
            CreateElementwiseOneRuntimeOneScalar(creation_context, op_def,
                                                 op_type, *scalar);
        *gpu_op = absl::make_unique<ElementwiseOneRuntimeOneScalar>(
            std::move(operation));
        return absl::OkStatus();
      } else if (linear_tensor) {
        ElementwiseTwoInput operation;
        RETURN_IF_ERROR(CreateElementwiseTwoInput(
            creation_context, op_def, op_type, *linear_tensor, &operation));
        *gpu_op = absl::make_unique<ElementwiseTwoInput>(std::move(operation));
        return absl::OkStatus();
      } else {
        if (inputs.size() == 2 &&
            (inputs[0]->tensor.shape.c == inputs[1]->tensor.shape.c ||
             inputs[1]->tensor.shape.c == 1)) {
          ElementwiseTwoInput operation = CreateElementwiseTwoInput(
              op_def, op_type, inputs[1]->tensor.shape);
          *gpu_op =
              absl::make_unique<ElementwiseTwoInput>(std::move(operation));
          return absl::OkStatus();
        } else if (inputs.size() >= 2) {
          auto output = outputs[0];
          std::vector<int> channels(inputs.size());
          for (int i = 0; i < inputs.size(); ++i) {
            channels[i] = inputs[i]->tensor.shape.c;
          }
          SelectAdd(op_def, channels, output->tensor.shape.c, gpu_op);
          return absl::OkStatus();
        }
        return absl::UnimplementedError(absl::StrCat(
            "No support of ", node.operation.type, " with this parameters"));
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
      if (inputs.size() == 1) {
        if (WinogradFromNode(creation_context, op_def, hints, input_shape,
                             output_shape, attr, gpu_subgraph)
                .ok()) {
          return absl::OkStatus();
        } else {
          gpu_op = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
          return SelectConvolution(attr, output_shape, creation_context, op_def,
                                   hints, gpu_op);
        }
      } else {
        auto weights_shape = inputs[1]->tensor.shape;
        TensorDescriptor weights_desc = {op_def.src_tensors[1].data_type,
                                         TensorStorageType::BUFFER,
                                         Layout::UNKNOWN};
        gpu_subgraph->operations.clear();
        gpu_subgraph->operations.resize(2);
        auto& converter_op = gpu_subgraph->operations[0];
        auto& conv_op = gpu_subgraph->operations[1];
        conv_op.input_ids = {0, -1};
        conv_op.output_ids = {0};
        OperationDef conv_def = op_def;
        conv_def.src_tensors[1] = weights_desc;
        ConvWeightsDescription conv_weights_desc;
        RETURN_IF_ERROR(SelectConvolutionWithDynamicWeights(
            attr, weights_shape, output_shape, creation_context, conv_def,
            hints, &conv_op.operation, &conv_weights_desc));

        int aligned_output =
            AlignByN(weights_shape.b, conv_weights_desc.output_group_size * 4);
        int aligned_input = AlignByN(weights_shape.c, 4);
        gpu_subgraph->new_tensors = {
            {BHWC(1, 1, 1,
                  aligned_output * aligned_input * weights_shape.h *
                      weights_shape.w),
             weights_desc}};
        OperationDef converter_def;
        converter_def.precision = op_def.precision;
        converter_def.src_tensors.push_back(op_def.src_tensors[1]);
        converter_def.dst_tensors.push_back(weights_desc);

        converter_op.input_ids = {1};
        converter_op.output_ids = {-1};
        return SelectConverterToConvWeights(conv_weights_desc, creation_context,
                                            converter_def, hints,
                                            &converter_op.operation);
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
      return absl::OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      auto attr =
          absl::any_cast<MaxUnpooling2DAttributes>(node.operation.attributes);
      SelectMaxUnpooling(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::MEAN: {
      auto attr = absl::any_cast<MeanAttributes>(node.operation.attributes);
      return SelectMean(attr, op_def, gpu_op);
    }
    case OperationType::MUL: {
      auto attr = absl::any_cast<MultiplyAttributes>(node.operation.attributes);
      const float* scalar = absl::get_if<float>(&attr.param);
      const auto* linear_tensor =
          absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
              &attr.param);
      if (scalar) {
        ElementwiseOneRuntimeOneScalar operation =
            CreateElementwiseOneRuntimeOneScalar(creation_context, op_def,
                                                 op_type, *scalar);
        *gpu_op = absl::make_unique<ElementwiseOneRuntimeOneScalar>(
            std::move(operation));
        return absl::OkStatus();
      } else if (linear_tensor) {
        ElementwiseTwoInput operation;
        RETURN_IF_ERROR(CreateElementwiseTwoInput(
            creation_context, op_def, op_type, *linear_tensor, &operation));
        *gpu_op = absl::make_unique<ElementwiseTwoInput>(std::move(operation));
        return absl::OkStatus();
      } else {
        if (inputs.size() == 2) {
          ElementwiseTwoInput operation = CreateElementwiseTwoInput(
              op_def, op_type, inputs[1]->tensor.shape);
          *gpu_op =
              absl::make_unique<ElementwiseTwoInput>(std::move(operation));
          return absl::OkStatus();
        } else {
          return absl::UnimplementedError(absl::StrCat(
              "No support of ", node.operation.type, " with this parameters"));
        }
      }
    }
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
      SelectPadding(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::POOLING_2D: {
      auto attr =
          absl::any_cast<Pooling2DAttributes>(node.operation.attributes);
      SelectPooling(attr, op_def, gpu_op);
      return absl::OkStatus();
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
      return absl::OkStatus();
    }
    case OperationType::RESHAPE: {
      const int src_channels = inputs[0]->tensor.shape.c;
      auto attr = absl::any_cast<ReshapeAttributes>(node.operation.attributes);
      SelectReshape(src_channels, attr.new_shape.c, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::RESIZE: {
      auto attr = absl::any_cast<Resize2DAttributes>(node.operation.attributes);
      return SelectResize(attr, op_def, gpu_op);
    }
    case OperationType::SLICE: {
      auto attr = absl::any_cast<SliceAttributes>(node.operation.attributes);
      SelectStridedSlice(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::SOFTMAX: {
      SelectSoftmax(inputs[0]->tensor.shape, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::SPACE_TO_DEPTH: {
      auto attr =
          absl::any_cast<SpaceToDepthAttributes>(node.operation.attributes);
      SelectSpaceToDepth(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::TRANSPOSE: {
      auto attr =
          absl::any_cast<TransposeAttributes>(node.operation.attributes);
      SelectTranspose(attr, op_def, gpu_op);
      return absl::OkStatus();
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
      return absl::OkStatus();
    }
    case OperationType::DIV:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      auto attr =
          absl::any_cast<ElementwiseAttributes>(node.operation.attributes);
      const float* scalar = absl::get_if<float>(&attr.param);
      const auto* linear_tensor =
          absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
              &attr.param);
      if (scalar) {
        ElementwiseOneRuntimeOneScalar operation =
            CreateElementwiseOneRuntimeOneScalar(creation_context, op_def,
                                                 op_type, *scalar);
        *gpu_op = absl::make_unique<ElementwiseOneRuntimeOneScalar>(
            std::move(operation));
        return absl::OkStatus();
      } else if (linear_tensor) {
        ElementwiseTwoInput operation;
        RETURN_IF_ERROR(CreateElementwiseTwoInput(
            creation_context, op_def, op_type, *linear_tensor, &operation));
        *gpu_op = absl::make_unique<ElementwiseTwoInput>(std::move(operation));
        return absl::OkStatus();
      } else {
        if (inputs.size() == 2) {
          ElementwiseTwoInput operation = CreateElementwiseTwoInput(
              op_def, op_type, inputs[1]->tensor.shape);
          *gpu_op =
              absl::make_unique<ElementwiseTwoInput>(std::move(operation));
          return absl::OkStatus();
        } else {
          return absl::UnimplementedError(absl::StrCat(
              "No support of ", node.operation.type, " with this parameters"));
        }
      }
    }
    default:
      return SelectDefault(creation_context, op_def, hints, inputs, outputs,
                           node, gpu_subgraph);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
