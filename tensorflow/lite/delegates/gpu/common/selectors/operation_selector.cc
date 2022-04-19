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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/flops_util.h"
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
  int min_src_depth = 16;
  int min_dst_depth = 16;
  if (gpu_info.IsAdreno()) {
    min_src_depth = 32;
    min_dst_depth = 32;
  } else if (gpu_info.IsAMD()) {
    min_dst_depth = 8;
  }
  int min_tiles = 32;
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno6xx()) {
      min_tiles = 128;
    } else {
      min_tiles = 64;
    }
  }
  const bool recommended_channels =
      src_depth >= min_src_depth && dst_depth >= min_dst_depth;
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
  winograd_up.name = "winograd_4x4_to_36";

  OperationDef conv_def;
  conv_def.precision = op_def.precision;
  conv_def.src_tensors.push_back(td_0);
  conv_def.dst_tensors.push_back(td_1);
  auto& conv = gpu_subgraph->operations[1];
  conv.input_ids = {-1};
  conv.output_ids = {-2};
  conv.operation = SelectConvolutionForWinograd(attr, input_shape, gpu_info,
                                                conv_def, hints);
  conv.name = "convolution_winograd_4x4_6x6";
  conv.operation->flops_ =
      GetConvolutionWinograd4x4To6x6Flops(output_shape, attr.weights.shape);

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
  winograd_down.name = "winograd_36_to_4x4";
  return absl::OkStatus();
}

// Supported operation types:
// 1) BATCHED_MATMUL
// 2) CONVOLUTION_2D
// 3) CONVOLUTION_TRANSPOSED
absl::Status AddDynamicConv(ModelHints hints, const GpuInfo& gpu_info,
                            const OperationDef& op_def, OperationType op_type,
                            const BHWC& src_shape, const OHWI& weights_shape,
                            const BHWC& dst_shape, int src_id, int weights_id,
                            int dst_id, GPUOperationsSubgraph* gpu_subgraph,
                            void* attr) {
  gpu_subgraph->operations.reserve(gpu_subgraph->operations.size() + 2);
  gpu_subgraph->operations.push_back({});
  auto& converter_op = gpu_subgraph->operations.back();
  gpu_subgraph->operations.push_back({});
  auto& conv_op = gpu_subgraph->operations.back();
  OperationDef conv_temp_def = op_def;
  conv_temp_def.src_tensors[1] = {op_def.src_tensors[1].data_type,
                                  TensorStorageType::BUFFER, Layout::HWC};
  WeightsDescription weights_desc;
  const BHWC weights_shape_bhwc(weights_shape.o, weights_shape.h,
                                weights_shape.w, weights_shape.i);
  conv_op.output_ids = {dst_id};
  if (op_type == OperationType::CONVOLUTION_2D) {
    Convolution2DAttributes* conv_attr =
        reinterpret_cast<Convolution2DAttributes*>(attr);
    conv_op.operation = SelectConvolutionWithDynamicWeights(
        *conv_attr, weights_shape_bhwc, dst_shape, gpu_info, conv_temp_def,
        hints, &weights_desc);
    conv_op.name = "convolution_dynamic";
    conv_op.operation->flops_ = GetConvolutionFlops(dst_shape, weights_shape);
  } else if (op_type == OperationType::CONVOLUTION_TRANSPOSED) {
    ConvolutionTransposedAttributes* conv_attr =
        reinterpret_cast<ConvolutionTransposedAttributes*>(attr);
    conv_op.operation = SelectConvolutionTransposedWithDynamicWeights(
        *conv_attr, gpu_info, conv_temp_def, &weights_desc);
    conv_op.name = "conv_transposed_dynamic";
    conv_op.operation->flops_ =
        GetConvolutionTransposedFlops(src_shape, weights_shape);
  } else if (op_type == OperationType::BATCHED_MATMUL) {
    Convolution2DAttributes* conv_attr =
        reinterpret_cast<Convolution2DAttributes*>(attr);
    conv_op.operation = SelectConvolutionWithDynamicWeights(
        *conv_attr, weights_shape_bhwc, dst_shape, gpu_info, conv_temp_def,
        hints, &weights_desc);
    conv_op.name = "mat_mul_as_convolution";
    conv_op.operation->flops_ = GetConvolutionFlops(dst_shape, weights_shape);
  } else {
    return absl::InternalError("No support of this operation type.");
  }
  int first_tensor_id = gpu_subgraph->new_tensors.size();
  if (weights_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weights_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    // weights are 4x textures 2d
    conv_op.input_ids = {src_id, -(first_tensor_id + 1), -(first_tensor_id + 2),
                         -(first_tensor_id + 3), -(first_tensor_id + 4)};
    uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
    for (int i = 0; i < 4; ++i) {
      gpu_subgraph->new_tensors.push_back(
          {BHWC(1, tex_size.y, tex_size.x, 4),
           TensorDescriptor(weights_desc.type, TensorStorageType::TEXTURE_2D,
                            Layout::HWC)});
    }
  } else {
    // weights are single buffer
    conv_op.input_ids = {src_id, -(first_tensor_id + 1)};
    gpu_subgraph->new_tensors.push_back(
        {BHWC(1, 1, 1,
              GetTotalElementsCountForLayout(weights_desc, weights_shape)),
         TensorDescriptor(weights_desc.type, TensorStorageType::BUFFER,
                          Layout::HWC)});
  }
  OperationDef conv_def = conv_op.operation->GetDefinition();
  OperationDef converter_def;
  converter_def.precision = op_def.precision;
  converter_def.src_tensors.push_back(op_def.src_tensors[1]);
  for (int i = 1; i < conv_def.src_tensors.size(); ++i) {
    converter_def.dst_tensors.push_back(conv_def.src_tensors[i]);
    converter_op.output_ids.push_back(-(first_tensor_id + i));
  }

  converter_op.input_ids = {weights_id};
  converter_op.operation =
      SelectConverterToConvWeights(weights_desc, converter_def, hints);
  converter_op.name = "bhwc_tensor_to_conv_weights";
  return absl::OkStatus();
}

void AddConvSharedWeights(
    const Convolution2DAttributes& attr, const WeightsDescription& weights_desc,
    std::vector<SharedWeightsConvDesc>* shared_conv_weights,
    GPUOperationsSubgraph* gpu_subgraph) {
  SharedWeightsConvDesc shared_weights_desc;
  shared_weights_desc.weights_id = attr.weights.id;
  shared_weights_desc.desc = weights_desc;
  int index = -1;
  for (int i = 0; i < shared_conv_weights->size(); ++i) {
    if ((*shared_conv_weights)[i] == shared_weights_desc) {
      index = i;
      break;
    }
  }
  if (index != -1) {
    const auto& new_ids = (*shared_conv_weights)[index].global_const_ids;
    for (int i = 0; i < new_ids.size(); ++i) {
      gpu_subgraph->operations[0].input_ids.push_back(new_ids[i]);
    }
  } else {
    shared_conv_weights->push_back(shared_weights_desc);
    if (weights_desc.layout ==
            WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
        weights_desc.layout ==
            WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
      // weights are 4x textures 2d
      uint2 tex_size = Get2dResourceSize(weights_desc, attr.weights.shape);
      const int flt_count =
          GetTotalElementsCountForLayout(weights_desc, attr.weights.shape);

      std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
      RearrangeWeights(attr.weights, weights_desc,
                       absl::MakeSpan(weights_data));
      int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
      for (int i = 0; i < 4; ++i) {
        TensorDescriptor weights_tensor = TensorDescriptor(
            weights_desc.type, TensorStorageType::TEXTURE_2D, Layout::HWC);
        weights_tensor.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
        weights_tensor.SetData(std::vector<uint8_t>(
            weights_data.data() + sub_size * i,
            weights_data.data() + sub_size * i + sub_size));
        gpu_subgraph->new_tensors.push_back(
            {weights_tensor.GetBHWCShape(), std::move(weights_tensor)});
        gpu_subgraph->operations[0].input_ids.push_back(-1 - i);
        shared_conv_weights->back().global_const_ids.push_back(-1 - i);
      }
    } else {
      // weights are single buffer
      TensorDescriptor weights_tensor = TensorDescriptor(
          weights_desc.type, TensorStorageType::BUFFER, Layout::HWC);
      const int flt_count =
          GetTotalElementsCountForLayout(weights_desc, attr.weights.shape);
      weights_tensor.SetBHWCShape(BHWC(1, 1, 1, flt_count));
      std::vector<uint8_t> weights_data =
          std::vector<uint8_t>(flt_count * SizeOf(weights_desc.type));
      RearrangeWeights(attr.weights, weights_desc,
                       absl::MakeSpan(weights_data));
      weights_tensor.SetData(std::move(weights_data));
      gpu_subgraph->new_tensors.push_back(
          {weights_tensor.GetBHWCShape(), std::move(weights_tensor)});
      gpu_subgraph->operations[0].input_ids.push_back(-1);
      shared_conv_weights->back().global_const_ids.push_back(-1);
    }
  }
}

}  // namespace

absl::Status GPUOperationFromNodePart0(
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints,
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    const Node& node, std::vector<SharedWeightsConvDesc>* shared_conv_weights,
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
      //   1) Transpose second tensor(weights). (1xBxHxW)->(Wx1xBxH)
      //   2) Run cconvolution with runtime weights
      auto second_shape = inputs[1]->tensor.shape;
      auto dst_shape = outputs[0]->tensor.shape;
      if (dst_shape.b != 1) {
        return absl::UnimplementedError(
            "Currently only batch = 1 supported for BATCHED_MATMUL.");
      }
      const OHWI weights_shape(second_shape.c, 1, second_shape.h,
                               second_shape.w);
      const BHWC weights_shape_bhwc(weights_shape.o, weights_shape.h,
                                    weights_shape.w, weights_shape.i);
      Convolution2DAttributes attr;
      attr.strides = HW(1, 1);
      attr.dilations = HW(1, 1);
      attr.padding.appended = HW(0, 0);
      attr.padding.prepended = HW(0, 0);
      attr.bias.shape = Linear(weights_shape.o);
      attr.bias.data.resize(weights_shape.o, 0.0f);

      gpu_subgraph->operations.clear();
      TensorDescriptor transposed_desc = {op_def.src_tensors[1].data_type,
                                          op_def.src_tensors[1].storage_type,
                                          Layout::BHWC};
      RETURN_IF_ERROR(SelectBestStorageType(
          gpu_info, weights_shape_bhwc, transposed_desc.storage_type,
          transposed_desc.data_type, transposed_desc.layout,
          &transposed_desc.storage_type));
      gpu_subgraph->operations.resize(1);
      auto& transpose_op = gpu_subgraph->operations[0];
      OperationDef transpose_def;
      transpose_def.precision = op_def.precision;
      transpose_def.src_tensors.push_back(op_def.src_tensors[1]);
      transpose_def.dst_tensors.push_back(transposed_desc);

      transpose_op.input_ids = {static_cast<int>(inputs[1]->id)};
      TransposeAttributes transpose_attr;
      transpose_attr.perm = BHWC(3, 0, 1, 2);
      transpose_op.operation = absl::make_unique<GPUOperation>(
          CreateTranspose(transpose_def, transpose_attr));
      transpose_op.name = "mat_mul_transpose_second_tensor";

      gpu_subgraph->new_tensors.push_back(
          {weights_shape_bhwc, transposed_desc});
      transpose_op.output_ids = {-1};

      OperationDef conv_def = op_def;
      conv_def.src_tensors[1] = transposed_desc;
      return AddDynamicConv(hints, gpu_info, conv_def, op_type,
                            inputs[0]->tensor.shape, weights_shape, dst_shape,
                            inputs[0]->id, -1, outputs[0]->id, gpu_subgraph,
                            &attr);
    }
    case OperationType::CAST:
      SelectCast(op_def, gpu_info, gpu_op);
      return absl::OkStatus();
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
          if (attr.groups != 1) {
            gpu_subgraph->operations[0].name = "convolution_2d_grouped";
          }
          if (!shared_conv_weights || attr.weights.id == -1) {
            *gpu_op =
                SelectConvolution(attr, output_shape, gpu_info, op_def, hints);
          } else {
            // Using convolutions with shared weights
            WeightsDescription weights_desc;
            const BHWC weights_shape_bhwc(
                attr.weights.shape.o, attr.weights.shape.h,
                attr.weights.shape.w, attr.weights.shape.i);
            OperationDef conv_temp_def = op_def;
            conv_temp_def.src_tensors.push_back(
                {op_def.src_tensors[0].data_type, TensorStorageType::BUFFER,
                 Layout::HWC});
            *gpu_op = SelectConvolutionWithDynamicWeights(
                attr, weights_shape_bhwc, output_shape, gpu_info, conv_temp_def,
                hints, &weights_desc);
            AddConvSharedWeights(attr, weights_desc, shared_conv_weights,
                                 gpu_subgraph);
          }
          (*gpu_op)->flops_ =
              GetConvolutionFlops(output_shape, attr.weights.shape);
          return absl::OkStatus();
        }
      } else {
        // CONVOLUTION_2D with runtime weights
        const OHWI weights_shape =
            OHWI(inputs[1]->tensor.shape.b, inputs[1]->tensor.shape.h,
                 inputs[1]->tensor.shape.w, inputs[1]->tensor.shape.c);
        if (weights_shape.i != inputs[0]->tensor.shape.c) {
          return absl::UnimplementedError(
              "No support of grouped convolution with runtime weights");
        }
        if (attr.bias.data.empty()) {
          attr.bias.shape = Linear(weights_shape.o);
          attr.bias.data.resize(weights_shape.o, 0.0f);
        }
        gpu_subgraph->operations.clear();
        return AddDynamicConv(hints, gpu_info, op_def, op_type, input_shape,
                              weights_shape, output_shape, inputs[0]->id,
                              inputs[1]->id, outputs[0]->id, gpu_subgraph,
                              &attr);
      }
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      auto attr = absl::any_cast<ConvolutionTransposedAttributes>(
          node.operation.attributes);
      if (inputs.size() == 1) {
        *gpu_op = SelectConvolutionTransposed(attr, gpu_info, op_def);
        (*gpu_op)->flops_ = GetConvolutionTransposedFlops(
            inputs[0]->tensor.shape, attr.weights.shape);
        return absl::OkStatus();
      } else {
        // CONVOLUTION_TRANSPOSED with runtime weights
        const OHWI weights_shape =
            OHWI(inputs[1]->tensor.shape.b, inputs[1]->tensor.shape.h,
                 inputs[1]->tensor.shape.w, inputs[1]->tensor.shape.c);
        if (attr.bias.data.empty()) {
          attr.bias.shape = Linear(weights_shape.o);
          attr.bias.data.resize(weights_shape.o, 0.0f);
        }
        gpu_subgraph->operations.clear();
        return AddDynamicConv(
            hints, gpu_info, op_def, op_type, inputs[0]->tensor.shape,
            weights_shape, outputs[0]->tensor.shape, inputs[0]->id,
            inputs[1]->id, outputs[0]->id, gpu_subgraph, &attr);
      }
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      auto attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
          node.operation.attributes);
      if (inputs.size() == 1) {
        *gpu_op = SelectDWConvolution(attr, gpu_info, op_def);
        (*gpu_op)->flops_ = GetDepthwiseConvolutionFlops(
            outputs[0]->tensor.shape, attr.weights.shape);
      } else {
        if (inputs[1]->tensor.shape.b != 1) {
          return absl::UnimplementedError(
              "No support of depthwise runtime weights with channel multiplier "
              "!= 1");
        }
        *gpu_op = SelectDWConvolutionDynamicWeights(attr, gpu_info, op_def);
        (*gpu_op)->flops_ = GetDepthwiseConvolutionFlops(
            outputs[0]->tensor.shape,
            OHWI(inputs[1]->tensor.shape.b, inputs[1]->tensor.shape.h,
                 inputs[1]->tensor.shape.w, inputs[1]->tensor.shape.c));
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
      (*gpu_op)->flops_ =
          GetFullyConnectedFlops(outputs[0]->tensor.shape, attr.weights.shape);
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
      std::vector<int> channels;
      channels.reserve(outputs.size());
      for (const auto& output : outputs) {
        channels.push_back(output->tensor.shape.c);
      }
      auto attr = absl::any_cast<SplitAttributes>(node.operation.attributes);
      SelectSplit(attr, channels, op_def, gpu_op);
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

absl::Status GPUOperationFromNode(
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints,
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    const Node& node, std::vector<SharedWeightsConvDesc>* shared_conv_weights,
    GPUOperationsSubgraph* gpu_subgraph) {
  RETURN_IF_ERROR(GPUOperationFromNodePart0(gpu_info, op_def, hints, inputs,
                                            outputs, node, shared_conv_weights,
                                            gpu_subgraph));
  for (auto& gpu_op : gpu_subgraph->operations) {
    if (gpu_op.name.empty()) {
      gpu_op.name = node.operation.type + " " + std::to_string(node.id);
    } else {
      gpu_op.name += " " + std::to_string(node.id);
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
