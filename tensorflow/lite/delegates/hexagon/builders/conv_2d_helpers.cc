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
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/conv_2d_builder.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

constexpr uint8_t k8BitSignFlipConstant = 0x80;
// 1/1024 ~ 0.0009766 is a restriction set by Hexagon's kernels.
// TODO(b/151103818): Figure out a way to retrieve this constant reliably.
constexpr float kHexagonMinRelativeScale = 0.0009766f;
// Channel count to split depthwise convolution op.
// See Conv2dOpBuilder.should_split_dwconv_ for details.
constexpr int kDwConv5x5Filt2x2StrideChannelCount = 32;

}  // namespace

TfLiteStatus ProcessPerChannelQuantizedWeights(
    const TfLiteTensor& weights_tensor, TfLiteContext* context,
    float* weights_min, float* weights_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant) {
  if (!per_channel_quant) return kTfLiteError;
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);

  // Retrieve channel scales.
  per_channel_quant->num_scale_values = weights_quant_params->scale->size;
  // Normalize the scales as expected by Hexagon.
  per_channel_quant->scales_data = weights_quant_params->scale->data;
  std::vector<float> normalized_scales;
  normalized_scales.reserve(per_channel_quant->num_scale_values);
  float scale_max = 0.0;
  for (int i = 0; i < per_channel_quant->num_scale_values; ++i) {
    normalized_scales.push_back(per_channel_quant->scales_data[i]);
    if (per_channel_quant->scales_data[i] > scale_max) {
      scale_max = per_channel_quant->scales_data[i];
    }
  }
  if (scale_max == 0.0) {
    TF_LITE_KERNEL_LOG(context, "Scale max is zero for: %s",
                       weights_tensor.name);
    return kTfLiteError;
  }
  for (int i = 0; i < per_channel_quant->num_scale_values; ++i) {
    normalized_scales[i] =
        std::max(normalized_scales[i] / scale_max, kHexagonMinRelativeScale);
  }
  // Add node for channel scales data.
  const std::vector<int> scales_shape = {1, 1, 1,
                                         per_channel_quant->num_scale_values};
  per_channel_quant->channel_scales_node = graph_builder->AddConstNodeWithData(
      scales_shape.data(), reinterpret_cast<char*>(normalized_scales.data()),
      normalized_scales.size() * sizeof(normalized_scales[0]));
  if (per_channel_quant->splits) {
    // Split channel scales to 32 channel batches.
    const std::vector<int> sliced_scales_shape = {
        1, 1, 1, kDwConv5x5Filt2x2StrideChannelCount};
    for (auto i = 0; i < per_channel_quant->splits; ++i) {
      auto offset = kDwConv5x5Filt2x2StrideChannelCount * i;
      auto* node = graph_builder->AddConstNodeWithData(
          sliced_scales_shape.data(),
          reinterpret_cast<char*>(normalized_scales.data() + offset),
          kDwConv5x5Filt2x2StrideChannelCount * sizeof(normalized_scales[0]));
      per_channel_quant->channel_scales_nodes.push_back(node);
    }
  }
  *weights_min = -128 * scale_max;
  *weights_max = 127 * scale_max;
  return kTfLiteOk;
}

TfLiteStatus ProcessPerChannelQuantizedBias(
    const TfLiteTensor& data_tensor, const TfLiteTensor& bias_tensor,
    const int bias_tensor_idx, TfLiteContext* context, float* bias_min,
    float* bias_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant,
    std::vector<int>* preprocessed_bias_data, OpBuilder** bias_const_node) {
  const TfLiteAffineQuantization* input_quant_params =
      static_cast<const TfLiteAffineQuantization*>(
          data_tensor.quantization.params);
  const float input_scale = input_quant_params->scale->data[0];
  // Now dequantize bias values to float first, to adjust for the
  // normalization of channel scales.
  auto* bias_data = bias_tensor.data.i32;
  const int bias_size = NumElements(&bias_tensor);
  if (bias_size != per_channel_quant->num_scale_values) {
    TF_LITE_KERNEL_LOG(
        context, "Bias/channel scales number mismatch for bias tensor: %s",
        bias_tensor.name);
    return kTfLiteError;
  }
  std::vector<float> dequantized_bias;
  dequantized_bias.reserve(bias_size);
  for (int i = 0; i < bias_size; ++i) {
    const float dequantized_value =
        bias_data[i] * input_scale * per_channel_quant->scales_data[i];
    const float abs_dequantized_value = std::abs(dequantized_value);
    if (abs_dequantized_value > *bias_max) {
      *bias_max = abs_dequantized_value;
    }
    dequantized_bias.push_back(dequantized_value);
  }
  *bias_max = *bias_max * 8;
  *bias_min = -1 * *bias_max;
  // Now requantize the bias values to the new min/max values.
  preprocessed_bias_data->reserve(per_channel_quant->num_scale_values);
  for (int i = 0; i < bias_size; ++i) {
    preprocessed_bias_data->push_back(static_cast<int>(
        std::round(std::pow(2, 31) * (dequantized_bias[i] / *bias_max))));
  }
  // Add nodes for bias.
  const std::vector<int> bias_shape = {1, 1, 1, bias_size};
  auto* bias_data_node = graph_builder->AddConstNodeWithData(
      bias_shape.data(),
      reinterpret_cast<char*>(preprocessed_bias_data->data()),
      preprocessed_bias_data->size() * sizeof((*preprocessed_bias_data)[0]));
  if (bias_const_node) {
    *bias_const_node = bias_data_node;
  }
  graph_builder->AddTensorWithID(bias_tensor_idx, bias_data_node->GetID(), 0,
                                 /*overwrite=*/true);
  return kTfLiteOk;
}

void Conv2dOpBuilder::CheckShouldSplitDwConv(TfLiteType weights_type,
                                             int input_depth,
                                             bool is_per_channel_quant,
                                             int channel_multiplier) {
  const TfLiteDepthwiseConvParams* conv_params =
      reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
  int weights_height = weight_shape_[0];
  int weights_width = weight_shape_[1];
  // input_depth * channel_multiplier
  int weights_depth_size = input_depth * weight_shape_[3];
  if (op_node_.op_type == OP_DepthwiseSupernode_8x8p32to8 &&
      weights_type == kTfLiteInt8 &&
      // weight_shape_ is [fh,fw,din,dmul]
      weights_height == 5 && weights_width == 5 &&
      // Stride larger than 2x2
      conv_params->stride_height >= 2 && conv_params->stride_width >= 2 &&
      // Depth more than 32 and is multiples of 32 so can be split.
      input_depth > kDwConv5x5Filt2x2StrideChannelCount &&
      input_depth % kDwConv5x5Filt2x2StrideChannelCount == 0 &&
      is_per_channel_quant && channel_multiplier == 1) {
    should_split_dwconv_ = true;
    // Splits the inputs to 32 channel batches.
    per_channel_quant_.splits =
        weights_depth_size / kDwConv5x5Filt2x2StrideChannelCount;
    per_channel_quant_.channel_scales_nodes.reserve(per_channel_quant_.splits);
  }
}

void Conv2dOpBuilder::SplitWeightsForDwConv(
    const std::vector<uint8_t>& converted_data, int input_depth,
    int channel_multiplier) {
  int weights_height_size = weight_shape_[0];
  int weights_width_size = weight_shape_[1];
  // Split the weight tensor into 32 channel batches.
  SplitParams split_params{
      .num_split = static_cast<uint16_t>(per_channel_quant_.splits),
      .axis = 2,
  };
  std::vector<RuntimeShape> split_shapes;
  std::vector<const tflite::RuntimeShape*> split_shapes_data;
  std::vector<std::vector<uint8_t>> splitted_weights;
  std::vector<uint8_t*> splitted_weights_data;
  split_shapes.reserve(per_channel_quant_.splits);
  split_shapes_data.reserve(per_channel_quant_.splits);
  splitted_weights.reserve(per_channel_quant_.splits);
  splitted_weights_data.reserve(per_channel_quant_.splits);
  for (auto s = 0; s < per_channel_quant_.splits; s++) {
    split_shapes.push_back({weights_height_size, weights_width_size,
                            kDwConv5x5Filt2x2StrideChannelCount,
                            channel_multiplier});
    split_shapes_data.push_back(&split_shapes.back());
    splitted_weights.emplace_back(weights_height_size * weights_width_size *
                                      channel_multiplier *
                                      kDwConv5x5Filt2x2StrideChannelCount,
                                  0);
    splitted_weights_data.push_back(splitted_weights.back().data());
  }
  RuntimeShape weight_shape = {weights_height_size, weights_width_size,
                               input_depth, channel_multiplier};
  optimized_ops::Split(split_params, weight_shape, converted_data.data(),
                       split_shapes_data.data(), splitted_weights_data.data());
  for (auto s = 0; s < per_channel_quant_.splits; s++) {
    auto splitted_weights_node = graph_builder_->AddConstNodeWithData(
        split_shapes[s].DimsData(),
        reinterpret_cast<char*>(splitted_weights_data[s]),
        splitted_weights[s].size() * sizeof(splitted_weights_data[s][0]));
    weights_nodes_.push_back(splitted_weights_node);
  }
}

TfLiteStatus Conv2dOpBuilder::InitializeWeightsNodes(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context, const int input_depth) {
  const std::vector<int> quant_bound_shape = {1, 1, 1, 1};

  const auto& weights_tensor = context->tensors[inputs->data[1]];
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    TF_LITE_KERNEL_LOG(
        context, "Weights tensor doesn't have correct allocation type: %s",
        weights_tensor.name);
    return kTfLiteError;
  }
  int weights_batch_size, weights_height_size, weights_width_size,
      weights_depth_size;
  // Hexagon lib expects the weight tensor in HWCN, TFLite uses NHWC.
  // Transpose NHWC -> HWCN
  GetDims(&weights_batch_size, &weights_height_size, &weights_width_size,
          &weights_depth_size, weights_tensor.dims);

  // Weights tensor could be int8 even for per-tensor quantization.
  // Therefore, we look at the number of scale values to check if it is
  // per-channel quantized.
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);
  const bool is_per_channel_quant = weights_quant_params->scale->size > 1;

  // WEIGHTS DATA.
  OpBuilder* weights_data_node = nullptr;
  if (op_node_.op_type == OP_Supernode_8x8p32to8) {
    // Hexagon lib expects the weight tensor in HWCN, TFLite uses NHWC.
    // Transpose NHWC -> HWCN
    weight_shape_ = {weights_height_size, weights_width_size,
                     weights_depth_size, weights_batch_size};
    RuntimeShape nhwc_shape({weights_batch_size, weights_height_size,
                             weights_width_size, weights_depth_size});
    RuntimeShape hwcn_shape({weights_height_size, weights_width_size,
                             weights_depth_size, weights_batch_size});
    std::vector<uint8_t> hwcn(NumElements(&weights_tensor));
    TransposeParams transpose_params;
    transpose_params.perm_count = 4;
    transpose_params.perm[0] = 1;
    transpose_params.perm[1] = 2;
    transpose_params.perm[2] = 3;
    transpose_params.perm[3] = 0;
    // TODO(b/151103818): Try merging Transpose & bit flip.
    if (weights_tensor.type == kTfLiteInt8) {
      optimized_ops::Transpose<int8_t>(transpose_params, nhwc_shape,
                                       weights_tensor.data.int8, hwcn_shape,
                                       reinterpret_cast<int8_t*>(hwcn.data()));
      // Flip bits on the weight values so that the int8 values are treated
      // as uint8.
      for (int i = 0; i < hwcn.size(); ++i) {
        hwcn[i] = hwcn[i] ^ k8BitSignFlipConstant;
      }
    } else {
      optimized_ops::Transpose<uint8_t>(transpose_params, nhwc_shape,
                                        weights_tensor.data.uint8, hwcn_shape,
                                        hwcn.data());
    }
    weights_data_node = graph_builder_->AddConstNodeWithData(
        weight_shape_.data(), reinterpret_cast<char*>(hwcn.data()),
        hwcn.size() * sizeof(hwcn[0]));
  } else if (op_node_.op_type == OP_DepthwiseSupernode_8x8p32to8) {
    // Hexagon treats depthwise conv like tf.nn.depthwise_conv2d, where the
    // expected filter shape is [fh,fw,din,dmul].
    // The data itself will remain the same, since TFLite's representation is
    // just a 'flattening' of Hexagon's version.
    const int channel_multiplier = weights_depth_size / input_depth;
    weight_shape_ = {weights_height_size, weights_width_size, input_depth,
                     channel_multiplier};

    // Check if the op hits the Depthwise conv accuracy issue.
    // See Conv2dOpBuilder.should_split_dwconv_ for details.
    CheckShouldSplitDwConv(weights_tensor.type, input_depth,
                           is_per_channel_quant, channel_multiplier);

    if (weights_tensor.type == kTfLiteInt8) {
      // Flip bits on the weight values so that the int8 values are treated
      // as uint8.
      std::vector<uint8_t> converted_data(NumElements(&weights_tensor));
      for (int i = 0; i < converted_data.size(); ++i) {
        converted_data[i] = weights_tensor.data.int8[i] ^ k8BitSignFlipConstant;
      }
      weights_data_node = graph_builder_->AddConstNodeWithData(
          weight_shape_.data(), reinterpret_cast<char*>(converted_data.data()),
          converted_data.size() * sizeof(converted_data[0]));
      if (should_split_dwconv_)
        SplitWeightsForDwConv(converted_data, input_depth, channel_multiplier);
    } else {
      weights_data_node = graph_builder_->AddConstNodeWithData(
          weight_shape_.data(), weights_tensor.data.raw,
          NumElements(&weights_tensor) * sizeof(weights_tensor.data.uint8[0]));
    }
  }
  graph_builder_->AddTensorWithID(inputs->data[1], weights_data_node->GetID(),
                                  0, /*overwrite=*/true);

  // WEIGHTS QUANTIZATION.
  float weights_min = 0;
  float weights_max = 0;
  if (is_per_channel_quant) {
    ProcessPerChannelQuantizedWeights(weights_tensor, context, &weights_min,
                                      &weights_max, graph_builder_,
                                      &per_channel_quant_);
  } else {
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        weights_tensor, &weights_min, &weights_max));
  }
  weights_min_node_ = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&weights_min),
      sizeof(weights_min));
  weights_max_node_ = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&weights_max),
      sizeof(weights_max));

  return kTfLiteOk;
}

void Conv2dOpBuilder::SplitBiasForDwConv(
    std::vector<int>& preprocessed_bias_data) {
  // Splits bias to 32 channel batches.
  std::vector<int> bias_shape = {1, 1, 1, kDwConv5x5Filt2x2StrideChannelCount};
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    auto offset = kDwConv5x5Filt2x2StrideChannelCount * i;
    auto* bias_data_node = graph_builder_->AddConstNodeWithData(
        bias_shape.data(),
        reinterpret_cast<char*>(preprocessed_bias_data.data() + offset),
        kDwConv5x5Filt2x2StrideChannelCount *
            sizeof(preprocessed_bias_data[0]));
    bias_nodes_.push_back(bias_data_node);
  }
}

TfLiteStatus Conv2dOpBuilder::InitializeBiasNodes(const TfLiteIntArray* inputs,
                                                  const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  const std::vector<int> quant_bound_shape = {1, 1, 1, 1};

  const auto& bias_tensor = context->tensors[inputs->data[2]];

  float bias_min = 0;
  float bias_max = 0;
  if (per_channel_quant_.channel_scales_node != nullptr) {
    std::vector<int> preprocessed_bias_data;
    ProcessPerChannelQuantizedBias(
        context->tensors[inputs->data[0]], bias_tensor, inputs->data[2],
        context, &bias_min, &bias_max, graph_builder_, &per_channel_quant_,
        &preprocessed_bias_data);
    if (should_split_dwconv_) SplitBiasForDwConv(preprocessed_bias_data);
  } else {
    auto* bias_data_node =
        graph_builder_->AddConstNodeWithData(inputs->data[2], bias_tensor);
    graph_builder_->AddTensorWithID(inputs->data[2], bias_data_node->GetID(), 0,
                                    /*overwrite=*/true);
    TF_LITE_ENSURE_STATUS(
        ComputeMinAndMaxQuantValues(bias_tensor, &bias_min, &bias_max));
  }

  bias_min_node_ = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&bias_min),
      sizeof(bias_min));
  bias_max_node_ = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&bias_max),
      sizeof(bias_max));

  return kTfLiteOk;
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
