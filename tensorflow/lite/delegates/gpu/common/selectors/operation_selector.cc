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

#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"

#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/convolution_transposed_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/default_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/dw_convolution_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/fully_connected_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/transpose.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {
namespace {
bool IsRecommendedForWinograd4x4To6x6(const Convolution2DAttributes& attr,
                                      const GpuInfo& gpu_info,
                                      const BHWC& dst_shape) {
  const int tiles_x = DivideRoundUp(dst_shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_shape.h, 4);
  const int total_tiles = tiles_x * tiles_y;
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  int min_depth = 16;
  if (gpu_info.IsAdreno() || gpu_info.IsAMD()) {
    min_depth = 32;
  }
  int min_tiles = 32;
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno6xx()) {
      min_tiles = 128;
    } else {
      min_tiles = 64;
    }
  }
  if (gpu_info.IsAMD()) {
    min_tiles = 64;
  }
  const bool recommended_channels =
      src_depth >= min_depth && dst_depth >= min_depth;
  const bool recommended_hw = total_tiles >= min_tiles;
  return recommended_channels && recommended_hw;
}

absl::Status WinogradFromNode(const GpuInfo& gpu_info,
                              const std::vector<Value*>& inputs,
                              const std::vector<Value*>& outputs,
                              const OperationDef& op_def, ModelHints hints,
                              const BHWC& input_shape, const BHWC& output_shape,
                              const Convolution2DAttributes& attr,
                              GPUOperationsSubgraph* gpu_subgraph) {
  if (!IsSuitableForWinograd4x4To6x6(attr)) {
    return absl::UnimplementedError("No implementation for this case.");
  }
  if (!IsRecommendedForWinograd4x4To6x6(attr, gpu_info, output_shape)) {
    return absl::UnimplementedError("Not recommended for this case.");
  }

  const int tiles_x = DivideRoundUp(output_shape.w, 4);
  const int tiles_y = DivideRoundUp(output_shape.h, 4);
  const BHWC shape_0{input_shape.b, 36, tiles_x * tiles_y, input_shape.c};
  const BHWC shape_1{input_shape.b, 36, tiles_x * tiles_y, output_shape.c};
  TensorDescriptor td_0;
  RETURN_IF_ERROR(SelectBestStorageType(
      gpu_info, shape_0, op_def.src_tensors[0].storage_type,
      op_def.src_tensors[0].data_type, op_def.src_tensors[0].layout,
      &td_0.storage_type));
  td_0.data_type = op_def.src_tensors[0].data_type;
  td_0.layout = op_def.src_tensors[0].layout;
  TensorDescriptor td_1;
  RETURN_IF_ERROR(SelectBestStorageType(
      gpu_info, shape_1, op_def.src_tensors[0].storage_type,
      op_def.src_tensors[0].data_type, op_def.src_tensors[0].layout,
      &td_1.storage_type));
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
  winograd_up.operation =
      SelectWinograd4x4To36(gpu_info, attr.padding, winograd_up_def);
  winograd_up.input_ids = {static_cast<int>(inputs[0]->id)};
  winograd_up.output_ids = {-1};

  OperationDef conv_def;
  conv_def.precision = op_def.precision;
  conv_def.src_tensors.push_back(td_0);
  conv_def.dst_tensors.push_back(td_1);
  auto& conv = gpu_subgraph->operations[1];
  conv.input_ids = {-1};
  conv.output_ids = {-2};
  conv.operation = SelectConvolutionForWinograd(attr, input_shape, gpu_info,
                                                conv_def, hints);

  OperationDef winograd_down_def;
  winograd_down_def.precision = op_def.precision;
  winograd_down_def.src_tensors.push_back(td_1);
  winograd_down_def.dst_tensors.push_back(op_def.dst_tensors[0]);
  auto& winograd_down = gpu_subgraph->operations[2];
  winograd_down.input_ids = {-2};
  winograd_down.output_ids = {static_cast<int>(outputs[0]->id)};
  auto bias_copy = attr.bias;
  if (bias_copy.shape.v < attr.weights.shape.o) {
    bias_copy.shape = Linear(attr.weights.shape.o);
    bias_copy.data.resize(attr.weights.shape.o);
  }
  winograd_down.operation =
      SelectWinograd36To4x4(gpu_info, winograd_down_def, bias_copy);
  return absl::OkStatus();
}

}  // namespace

absl::Status GPUOperationFromNode(const GpuInfo& gpu_info,
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
      if (inputs.size() == 2 &&
          (inputs[0]->tensor.shape.c == inputs[1]->tensor.shape.c ||
           inputs[1]->tensor.shape.c == 1)) {
        GPUOperation operation =
            CreateElementwiseTwoInput(op_def, op_type, inputs[1]->tensor.shape);
        *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
        return absl::OkStatus();
      } else if (inputs.size() >= 2) {
        auto output = outputs[0];
        std::vector<int> channels(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
          channels[i] = inputs[i]->tensor.shape.c;
        }
        SelectAdd(op_def, channels, output->tensor.shape.c, gpu_op);
        return absl::OkStatus();
      } else if (inputs.size() == 1 && node.operation.attributes.has_value()) {
        auto attr =
            absl::any_cast<ElementwiseAttributes>(node.operation.attributes);
        GPUOperation operation =
            CreateElementwise(gpu_info, op_def, op_type, attr);
        *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
        return absl::OkStatus();
      }
      return absl::UnimplementedError(absl::StrCat(
          "No support of ", node.operation.type, " with this parameters"));
    }
    case OperationType::BATCHED_MATMUL: {
      // Currently only batch = 1 is supported.
      // Matmul replaced with this sequence:
      //   1) Transpose second tensor(weights). (1x1xHxW)->(Wx1x1xH)
      //   2) Convert second tensor(weights) from 1) to Convolution weights
      //   3) Run usual convolution
      auto second_shape = inputs[1]->tensor.shape;
      auto dst_shape = outputs[0]->tensor.shape;
      if (dst_shape.b != 1) {
        return absl::UnimplementedError(
            "Currently only batch = 1 supported for BATCHED_MATMUL.");
      }
      BHWC weights_shape(second_shape.c, 1, 1, second_shape.w);
      Convolution2DAttributes attr;
      attr.strides = HW(1, 1);
      attr.dilations = HW(1, 1);
      attr.padding.appended = HW(0, 0);
      attr.padding.prepended = HW(0, 0);
      attr.bias.shape = Linear(weights_shape.b);
      attr.bias.data.resize(weights_shape.b, 0.0f);

      TensorDescriptor transposed_desc = {op_def.src_tensors[1].data_type,
                                          op_def.src_tensors[1].storage_type,
                                          Layout::BHWC};
      RETURN_IF_ERROR(SelectBestStorageType(
          gpu_info, weights_shape, transposed_desc.storage_type,
          transposed_desc.data_type, transposed_desc.layout,
          &transposed_desc.storage_type));
      TensorDescriptor weights_desc = {op_def.src_tensors[1].data_type,
                                       TensorStorageType::BUFFER, Layout::BHWC};
      gpu_subgraph->operations.clear();
      gpu_subgraph->operations.resize(3);
      auto& transpose_op = gpu_subgraph->operations[0];
      auto& converter_op = gpu_subgraph->operations[1];
      auto& conv_op = gpu_subgraph->operations[2];
      conv_op.input_ids = {static_cast<int>(inputs[0]->id), -1};
      conv_op.output_ids = {static_cast<int>(outputs[0]->id)};
      OperationDef conv_def = op_def;
      conv_def.src_tensors[1] = weights_desc;
      WeightsDescription conv_weights_desc;
      conv_op.operation = SelectConvolutionWithDynamicWeights(
          attr, weights_shape, dst_shape, gpu_info, conv_def, hints,
          &conv_weights_desc);

      int aligned_output =
          AlignByN(weights_shape.b, conv_weights_desc.GetOutputGroupSize() * 4);
      int aligned_input = AlignByN(weights_shape.c, 4);
      gpu_subgraph->new_tensors = {{BHWC(1, 1, 1,
                                         aligned_output * aligned_input *
                                             weights_shape.h * weights_shape.w),
                                    weights_desc},
                                   {weights_shape, transposed_desc}};
      OperationDef converter_def;
      converter_def.precision = op_def.precision;
      converter_def.src_tensors.push_back(transposed_desc);
      converter_def.dst_tensors.push_back(weights_desc);

      converter_op.input_ids = {-2};
      converter_op.output_ids = {-1};
      converter_op.operation =
          SelectConverterToConvWeights(conv_weights_desc, converter_def, hints);

      OperationDef transpose_def;
      transpose_def.precision = op_def.precision;
      transpose_def.src_tensors.push_back(op_def.src_tensors[1]);
      transpose_def.dst_tensors.push_back(transposed_desc);

      transpose_op.input_ids = {static_cast<int>(inputs[1]->id)};
      transpose_op.output_ids = {-2};
      TransposeAttributes transpose_attr;
      transpose_attr.perm = BHWC(3, 0, 1, 2);
      transpose_op.operation = absl::make_unique<GPUOperation>(
          CreateTranspose(transpose_def, transpose_attr));
      return absl::OkStatus();
    }
    case OperationType::CONCAT: {
      auto attr = absl::any_cast<ConcatAttributes>(node.operation.attributes);
      const int max_inputs = gpu_info.GetMaxImageArguments() - 8;
      if (inputs.size() >= max_inputs) {
        int groups = DivideRoundUp(inputs.size(), max_inputs);
        gpu_subgraph->operations.clear();
        gpu_subgraph->operations.resize(groups);
        BHWC concatenated_shape = inputs[0]->tensor.shape;
        concatenated_shape.set(attr.axis, 0);
        for (int g = 0; g < groups; ++g) {
          std::vector<int> channels;
          auto& concat_op = gpu_subgraph->operations[g];
          OperationDef new_def;
          new_def.precision = op_def.precision;
          if (g != 0) {
            // concatenated tensor from previos concats
            new_def.src_tensors.push_back(op_def.dst_tensors[0]);
            concat_op.input_ids = {-g};
            channels.push_back(concatenated_shape.c);
          }
          for (int i = 0; i < max_inputs; ++i) {
            int src_index = g * max_inputs + i;
            if (src_index >= op_def.src_tensors.size()) {
              break;
            }
            new_def.src_tensors.push_back(op_def.src_tensors[src_index]);
            concat_op.input_ids.push_back(inputs[src_index]->id);
            channels.push_back(inputs[src_index]->tensor.shape.c);
            int current_size = concatenated_shape.get(attr.axis);
            concatenated_shape.set(
                attr.axis,
                current_size + inputs[src_index]->tensor.shape.get(attr.axis));
          }
          new_def.dst_tensors.push_back(op_def.dst_tensors[0]);
          if (g == groups - 1) {
            // last concat
            concat_op.output_ids = {static_cast<int>(outputs[0]->id)};
          } else {
            // intermediate concat, create new tensor for it
            concat_op.output_ids = {-(g + 1)};
            gpu_subgraph->new_tensors.push_back(
                {concatenated_shape, op_def.dst_tensors[0]});
          }
          RETURN_IF_ERROR(SelectConcat(attr, channels, new_def, gpu_info,
                                       &concat_op.operation));
        }
        return absl::OkStatus();
      } else {
        std::vector<int> channels(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
          channels[i] = inputs[i]->tensor.shape.c;
        }
        return SelectConcat(attr, channels, op_def, gpu_info, gpu_op);
      }
    }
    case OperationType::CONVOLUTION_2D: {
      auto attr =
          absl::any_cast<Convolution2DAttributes>(node.operation.attributes);
      auto input_shape = inputs[0]->tensor.shape;
      auto output_shape = outputs[0]->tensor.shape;
      if (inputs.size() == 1) {
        if (!hints.Check(ModelHints::kNoWinogradOptimizations) &&
            WinogradFromNode(gpu_info, inputs, outputs, op_def, hints,
                             input_shape, output_shape, attr, gpu_subgraph)
                .ok()) {
          return absl::OkStatus();
        } else {
          gpu_op = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
          *gpu_op =
              SelectConvolution(attr, output_shape, gpu_info, op_def, hints);
          return absl::OkStatus();
        }
      } else {
        auto weights_shape = inputs[1]->tensor.shape;
        if (attr.bias.data.empty()) {
          attr.bias.shape = Linear(weights_shape.b);
          attr.bias.data.resize(weights_shape.b, 0.0f);
        }
        TensorDescriptor weights_desc = {op_def.src_tensors[1].data_type,
                                         TensorStorageType::BUFFER,
                                         Layout::BHWC};
        gpu_subgraph->operations.clear();
        gpu_subgraph->operations.resize(2);
        auto& converter_op = gpu_subgraph->operations[0];
        auto& conv_op = gpu_subgraph->operations[1];
        conv_op.input_ids = {static_cast<int>(inputs[0]->id), -1};
        conv_op.output_ids = {static_cast<int>(outputs[0]->id)};
        OperationDef conv_def = op_def;
        conv_def.src_tensors[1] = weights_desc;
        WeightsDescription conv_weights_desc;
        conv_op.operation = SelectConvolutionWithDynamicWeights(
            attr, weights_shape, output_shape, gpu_info, conv_def, hints,
            &conv_weights_desc);

        int aligned_output = AlignByN(
            weights_shape.b, conv_weights_desc.GetOutputGroupSize() * 4);
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

        converter_op.input_ids = {static_cast<int>(inputs[1]->id)};
        converter_op.output_ids = {-1};
        converter_op.operation = SelectConverterToConvWeights(
            conv_weights_desc, converter_def, hints);
        return absl::OkStatus();
      }
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      auto attr = absl::any_cast<ConvolutionTransposedAttributes>(
          node.operation.attributes);
      if (inputs.size() == 1) {
        *gpu_op = SelectConvolutionTransposed(attr, gpu_info, op_def);
        return absl::OkStatus();
      } else {
        // CONVOLUTION_TRANSPOSED with runtime weights
        OHWI weights_shape =
            OHWI(inputs[1]->tensor.shape.b, inputs[1]->tensor.shape.h,
                 inputs[1]->tensor.shape.w, inputs[1]->tensor.shape.c);
        if (attr.bias.data.empty()) {
          attr.bias.shape = Linear(weights_shape.o);
          attr.bias.data.resize(weights_shape.o, 0.0f);
        }
        gpu_subgraph->operations.clear();
        gpu_subgraph->operations.resize(2);
        auto& converter_op = gpu_subgraph->operations[0];
        auto& conv_op = gpu_subgraph->operations[1];
        WeightsDescription weights_desc;
        conv_op.operation = SelectConvolutionTransposedWithDynamicWeights(
            attr, gpu_info, op_def, &weights_desc);
        conv_op.output_ids = {static_cast<int>(outputs[0]->id)};

        const int dst_depth = AlignByN(DivideRoundUp(weights_shape.o, 4),
                                       weights_desc.GetOutputGroupSize());
        const int src_depth = DivideRoundUp(weights_shape.i, 4);
        const int kernel_x = weights_shape.w;
        const int kernel_y = weights_shape.h;
        if (weights_desc.layout ==
                WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
            weights_desc.layout ==
                WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
          // weights are 4x textures 2d
          conv_op.input_ids = {static_cast<int>(inputs[0]->id), -1, -2, -3, -4};
          int texture_width = dst_depth;
          int texture_height = src_depth * kernel_x * kernel_y;
          for (int i = 0; i < 4; ++i) {
            gpu_subgraph->new_tensors.push_back(
                {BHWC(1, texture_height, texture_width, 4),
                 TensorDescriptor(op_def.GetDataType(),
                                  TensorStorageType::TEXTURE_2D, Layout::HWC)});
          }
        } else {
          // weights is single buffer
          conv_op.input_ids = {static_cast<int>(inputs[0]->id), -1};
          gpu_subgraph->new_tensors = {
              {BHWC(
                   1, 1, 1,
                   GetTotalElementsCountForLayout(weights_desc, weights_shape)),
               TensorDescriptor(op_def.GetDataType(), TensorStorageType::BUFFER,
                                Layout::HWC)}};
        }
        OperationDef conv_def = conv_op.operation->GetDefinition();
        OperationDef converter_def;
        converter_def.precision = op_def.precision;
        converter_def.src_tensors.push_back(op_def.src_tensors[1]);
        for (int i = 1; i < conv_def.src_tensors.size(); ++i) {
          converter_def.dst_tensors.push_back(conv_def.src_tensors[i]);
          converter_op.output_ids.push_back(-i);
        }

        converter_op.input_ids = {static_cast<int>(inputs[1]->id)};
        converter_op.operation =
            SelectConverterToConvWeights(weights_desc, converter_def, hints);
        return absl::OkStatus();
      }
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      auto attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
          node.operation.attributes);
      if (inputs.size() == 1) {
        *gpu_op = SelectDWConvolution(attr, gpu_info, op_def);
      } else {
        if (inputs[1]->tensor.shape.b != 1) {
          return absl::UnimplementedError(
              "No support of depthwise runtime weights with channel multiplier "
              "!= 1");
        }
        *gpu_op = SelectDWConvolutionDynamicWeights(attr, gpu_info, op_def);
      }
      return absl::OkStatus();
    }
    case OperationType::DEPTH_TO_SPACE: {
      auto attr =
          absl::any_cast<SpaceToDepthAttributes>(node.operation.attributes);
      SelectDepthToSpace(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::FULLY_CONNECTED: {
      auto attr =
          absl::any_cast<FullyConnectedAttributes>(node.operation.attributes);
      *gpu_op = SelectFullyConnected(attr, gpu_info, op_def,
                                     inputs[0]->tensor.shape.b);
      return absl::OkStatus();
    }
    case OperationType::FULLY_CONNECTED_INT8: {
      auto attr = absl::any_cast<FullyConnectedInt8Attributes>(
          node.operation.attributes);
      *gpu_op = SelectFullyConnected(attr, gpu_info, op_def);
      return absl::OkStatus();
    }
    case OperationType::GATHER: {
      auto attr = absl::any_cast<GatherAttributes>(node.operation.attributes);
      RETURN_IF_ERROR(SelectGather(attr, op_def, gpu_op));
      return absl::OkStatus();
    }
    case OperationType::LSTM: {
      *gpu_op = SelectLSTM(op_def, gpu_info);
      return absl::OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      auto attr =
          absl::any_cast<MaxUnpooling2DAttributes>(node.operation.attributes);
      *gpu_op = SelectMaxUnpooling(attr, op_def);
      return absl::OkStatus();
    }
    case OperationType::MEAN: {
      auto attr = absl::any_cast<MeanAttributes>(node.operation.attributes);
      *gpu_op = SelectReduce(attr.dims, inputs[0]->tensor.shape, op_type,
                             op_def, gpu_info);
      return absl::OkStatus();
    }
    case OperationType::MEAN_STDDEV_NORMALIZATION: {
      MeanStdDevNormalization operation = CreateMeanStdDevNormalization(
          op_def, gpu_info, (inputs[0]->tensor.shape.c + 3) / 4);
      *gpu_op =
          absl::make_unique<MeanStdDevNormalization>(std::move(operation));
      return absl::OkStatus();
    }
    case OperationType::PAD: {
      auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
      SelectPadding(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::POOLING_2D: {
      auto attr =
          absl::any_cast<Pooling2DAttributes>(node.operation.attributes);
      *gpu_op = SelectPooling(attr, op_def);
      return absl::OkStatus();
    }
    case OperationType::PRELU: {
      auto attr = absl::any_cast<PReLUAttributes>(node.operation.attributes);
      *gpu_op = SelectPReLU(attr, gpu_info, op_def);
      return absl::OkStatus();
    }
    case OperationType::QUANTIZE_AND_DEQUANTIZE: {
      auto attr = absl::any_cast<QuantizeAndDequantizeAttributes>(
          node.operation.attributes);
      *gpu_op = SelectQuantizeAndDequantize(attr, op_def);
      return absl::OkStatus();
    }
    case OperationType::RELU: {
      auto attr = absl::any_cast<ReLUAttributes>(node.operation.attributes);
      *gpu_op = SelectReLU(attr, op_def);
      return absl::OkStatus();
    }
    case OperationType::RESAMPLER: {
      *gpu_op = SelectResampler(op_def);
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
    case OperationType::SPLIT: {
      auto attr = absl::any_cast<SplitAttributes>(node.operation.attributes);
      SelectSplit(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
    case OperationType::TILE: {
      *gpu_op = SelectTile(op_def, inputs[0]->tensor.shape);
      return absl::OkStatus();
    }
    case OperationType::TRANSPOSE: {
      auto attr =
          absl::any_cast<TransposeAttributes>(node.operation.attributes);
      SelectTranspose(attr, op_def, gpu_op);
      return absl::OkStatus();
    }
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
      GPUOperation operation =
          CreateElementwiseOneInput(gpu_info, op_def, op_type);
      *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    case OperationType::DIV:
    case OperationType::EQUAL:
    case OperationType::GREATER:
    case OperationType::GREATER_EQUAL:
    case OperationType::LESS:
    case OperationType::LESS_EQUAL:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::MUL:
    case OperationType::NOT_EQUAL:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      if (inputs.size() == 2) {
        GPUOperation operation =
            CreateElementwiseTwoInput(op_def, op_type, inputs[1]->tensor.shape);
        *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
        return absl::OkStatus();
      } else if (inputs.size() == 1 && node.operation.attributes.has_value()) {
        auto attr =
            absl::any_cast<ElementwiseAttributes>(node.operation.attributes);
        GPUOperation operation =
            CreateElementwise(gpu_info, op_def, op_type, attr);
        *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
        return absl::OkStatus();
      }
      return absl::UnimplementedError(absl::StrCat(
          "No support of ", node.operation.type, " with this parameters"));
    }
    case OperationType::REDUCE_MAXIMUM:
    case OperationType::REDUCE_MINIMUM:
    case OperationType::REDUCE_PRODUCT:
    case OperationType::REDUCE_SUM: {
      auto attr = absl::any_cast<ReduceAttributes>(node.operation.attributes);
      *gpu_op = SelectReduce(attr.dims, inputs[0]->tensor.shape, op_type,
                             op_def, gpu_info);
      return absl::OkStatus();
    }
    default:
      return SelectDefault(gpu_info, op_def, hints, inputs, outputs, node,
                           gpu_subgraph);
  }
}

}  // namespace gpu
}  // namespace tflite
