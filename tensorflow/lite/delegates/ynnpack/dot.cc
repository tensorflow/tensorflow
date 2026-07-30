/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/ynnpack/dot.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {

namespace {

TfLiteStatus DefineQuantizedDot(
    TfLiteContext* context, ynn_subgraph_t subgraph, int rank_a, int rank_b,
    absl::Span<const int32_t> a_reduce_axes,
    absl::Span<const int32_t> b_reduce_axes, uint32_t a_id, uint32_t a_scale_id,
    uint32_t a_zp_id, uint32_t b_id, uint32_t b_scale_id, uint32_t b_zp_id,
    uint32_t bias_id, int bias_rank, uint32_t out_scale_id, uint32_t out_zp_id,
    bool is_per_channel, bool is_conv, ynn_type output_ynn_type,
    uint32_t* output_id) {
  TF_LITE_ENSURE_EQ(context, a_reduce_axes.size(), b_reduce_axes.size());
  int num_k_dims = a_reduce_axes.size();
  // We assume a_id and b_id are quantized (int8 or uint8).
  // Accumulator will be int32.

  int rank = rank_a - num_k_dims + 1;
  bool is_dynamically_quantized = (out_scale_id == YNN_INVALID_VALUE_ID);

  // If grouped (rank_b == 5) and per-channel, expand b_scale and b_zp to rank 3
  // [G, 1, CO_pg].
  if (is_conv && rank_b == 5 && is_per_channel) {
    if (b_scale_id != YNN_INVALID_VALUE_ID) {
      uint32_t expanded_scale_id = YNN_INVALID_VALUE_ID;
      int32_t axes[] = {1};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
          subgraph, 1, axes, b_scale_id, &expanded_scale_id, 0));
      b_scale_id = expanded_scale_id;
    }
    if (b_zp_id != YNN_INVALID_VALUE_ID) {
      uint32_t expanded_zp_id = YNN_INVALID_VALUE_ID;
      int32_t axes[] = {1};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
          subgraph, 1, axes, b_zp_id, &expanded_zp_id, 0));
      b_zp_id = expanded_zp_id;
    }
  }

  // Compute zero point and scale of the dot product.
  uint32_t dot_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_dot_quantization(
      subgraph, num_k_dims, a_id, a_zp_id, a_scale_id, b_id, b_zp_id,
      b_scale_id, dot_zp_id, dot_scale_id));

  uint32_t accum_init_id = YNN_INVALID_VALUE_ID;

  if (dot_zp_id != YNN_INVALID_VALUE_ID) {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(subgraph, ynn_unary_negate,
                                               dot_zp_id, &accum_init_id, 0));
  }

  if (bias_id != YNN_INVALID_VALUE_ID) {
    if (accum_init_id == YNN_INVALID_VALUE_ID) {
      accum_init_id = bias_id;
    } else {
      uint32_t sub_id = YNN_INVALID_VALUE_ID;
      uint32_t broadcasted_bias_id = bias_id;
      TF_LITE_ENSURE_STATUS(
          ImplementMutualBroadcasting(context, subgraph, rank, bias_rank, 0, 0,
                                      accum_init_id, broadcasted_bias_id));
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_binary(subgraph, ynn_binary_add, accum_init_id,
                            broadcasted_bias_id, &sub_id, 0));
      accum_init_id = sub_id;
    }
  }

  // Now define the dot product.
  uint32_t accum_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, num_k_dims, a_id, b_id,
                                           accum_init_id, &accum_id, 0));

  uint32_t accum_scale_id = dot_scale_id;

  if (is_dynamically_quantized) {
    // Dequantize accumulator directly to output.
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                              accum_scale_id, ynn_type_fp32, output_id, 0));
  } else {
    // Dequantize accumulator to float.
    uint32_t float_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                              accum_scale_id, ynn_type_fp32, &float_id, 0));

    // Quantize back to output.
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(subgraph, float_id,
                                                  output_ynn_type, out_zp_id,
                                                  out_scale_id, output_id, 0));
  }

  return kTfLiteOk;
}

TfLiteStatus DefineMatMul(TfLiteContext* context, ynn_subgraph_t subgraph,
                          int rank_a, int rank_b, uint32_t a_id, uint32_t b_id,
                          uint32_t bias_id, bool adj_x, bool adj_y,
                          bool mutual_broadcast,
                          const TfLiteTensor& input_a_tensor,
                          const TfLiteTensor& input_b_tensor,
                          const TfLiteTensor& output_tensor,
                          uint32_t* output_id) {
  bool is_input_a_quantized = IsQuantized(input_a_tensor);
  bool is_input_b_quantized = IsQuantized(input_b_tensor);
  bool is_output_quantized = IsQuantized(output_tensor);

  bool is_quantized = is_input_a_quantized && is_output_quantized;
  bool is_dynamically_quantized =
      !is_input_a_quantized && is_input_b_quantized && !is_output_quantized;

  bool is_per_channel = false;
  if (is_quantized || is_dynamically_quantized) {
    const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
        input_b_tensor.quantization.params);
    is_per_channel =
        quant_params && quant_params->scale && quant_params->scale->size > 1;
  }

  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t out_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t out_zp_id = YNN_INVALID_VALUE_ID;

  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, input_b_tensor, &b_scale_id, &b_zp_id));
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, output_tensor, &out_scale_id, &out_zp_id));

  uint32_t current_a_id = a_id;
  uint32_t current_b_id = b_id;

  if (is_dynamically_quantized) {
    // 1. Reduce min_max. Last axis is K (axis -1 or -2 depending on adj_x).
    int32_t reduce_axis = adj_x ? -2 : -1;
    uint32_t min_max_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
        subgraph, ynn_reduce_min_max, 1, &reduce_axis, current_a_id,
        YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS));

    // 2. Define dynamic quantization params.
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dynamic_quantization(
        subgraph, min_max_id, ynn_type_int8, &a_zp_id, &a_scale_id, 0));

    // 3. Quantize input.
    uint32_t quantized_a_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_quantize(subgraph, current_a_id, ynn_type_int8, a_zp_id,
                            a_scale_id, &quantized_a_id, 0));

    current_a_id = quantized_a_id;
  } else {
    TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
        context, subgraph, input_a_tensor, &a_scale_id, &a_zp_id));
  }

  auto transpose = [&](int rank, uint32_t& val_id) -> TfLiteStatus {
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;
    int32_t perm[YNN_MAX_TENSOR_RANK];
    std::iota(perm, perm + rank, 0);
    std::swap(perm[rank - 1], perm[rank - 2]);

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, rank, perm, val_id, &transposed_id, 0));
    val_id = transposed_id;
    return kTfLiteOk;
  };

  if (adj_x) {
    TF_LITE_ENSURE_STATUS(transpose(rank_a, current_a_id));
    if (is_dynamically_quantized) {
      TF_LITE_ENSURE_STATUS(transpose(rank_a, a_zp_id));
      TF_LITE_ENSURE_STATUS(transpose(rank_a, a_scale_id));
    }
  }
  if (adj_y) {
    TF_LITE_ENSURE_STATUS(transpose(rank_b, current_b_id));
  }

  if (mutual_broadcast) {
    TF_LITE_ENSURE_STATUS(ImplementMutualBroadcasting(
        context, subgraph, rank_a, rank_b, /*exclude_a=*/2, /*exclude_b=*/2,
        current_a_id, current_b_id));
    rank_a = std::max(rank_a, rank_b);
    rank_b = rank_a;
  }

  // Broadcast bias if present (only for float and fully quantized cases).
  uint32_t broadcasted_bias_id = bias_id;

  if (is_quantized || is_dynamically_quantized) {
    uint32_t dot_output_id = *output_id;
    if (is_dynamically_quantized && bias_id != YNN_INVALID_VALUE_ID) {
      // We need intermediate output for dot before adding bias.
      dot_output_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(subgraph, ynn_type_fp32,
                                                  rank_a, nullptr, nullptr, 0,
                                                  &dot_output_id));
    }

    TF_LITE_ENSURE_STATUS(DefineQuantizedDot(
        context, subgraph, rank_a, rank_b, {rank_a - 1}, {rank_b - 2},
        current_a_id, a_scale_id, a_zp_id, current_b_id, b_scale_id, b_zp_id,
        is_dynamically_quantized ? YNN_INVALID_VALUE_ID : broadcasted_bias_id,
        rank_a - 1, out_scale_id, out_zp_id, is_per_channel, /*is_conv=*/false,
        GetYnnType(output_tensor.type), &dot_output_id));

    if (is_dynamically_quantized && bias_id != YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
          subgraph, ynn_binary_add, dot_output_id, bias_id, output_id, 0));
    } else {
      *output_id = dot_output_id;
    }
  } else {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dot(subgraph, /*num_k_dims=*/1, current_a_id, current_b_id,
                       broadcasted_bias_id, output_id, 0));
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteStatus IsBatchMatMulSupported(const TfLiteRegistration* registration,
                                    const TfLiteNode* node,
                                    TfLiteContext* context,
                                    bool is_runtime_bmm) {
  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input_a = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& input_b = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, tflite::NumElements(&input_a) > 0);
  TF_LITE_ENSURE(context, tflite::NumElements(&input_b) > 0);

  TF_LITE_ENSURE(context, IsTensorSupported(input_a));
  TF_LITE_ENSURE(context,
                 IsTensorSupported(input_b, /*allow_per_channel=*/true));
  TF_LITE_ENSURE(context, IsTensorSupported(output));

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  if (input_a.type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, input_b.type == kTfLiteInt8 ||
                                input_b.type == kTfLiteInt4 ||
                                input_b.type == kTfLiteUInt4 ||
                                input_b.type == kTfLiteInt2);
    TF_LITE_ENSURE(context,
                   output.type == kTfLiteInt8 || output.type == kTfLiteInt32);
  } else if (is_float_type(input_a.type)) {
    if (!is_float_type(input_b.type)) {
      TF_LITE_ENSURE(context, input_b.type == kTfLiteInt8 ||
                                  input_b.type == kTfLiteInt4 ||
                                  input_b.type == kTfLiteUInt4 ||
                                  input_b.type == kTfLiteInt2);
      if (!is_runtime_bmm) {
        const auto* params =
            static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);
        TF_LITE_ENSURE(context,
                       params != nullptr && params->asymmetric_quantize_inputs);
      } else {
        // odml.runtime_bmm just assumes that float x int is dynamically
        // quantized...?
      }
    }
    TF_LITE_ENSURE(context, is_float_type(output.type));
  } else {
    return kTfLiteError;
  }

  if (input_b.type == kTfLiteInt4 || input_b.type == kTfLiteUInt4) {
    TF_LITE_ENSURE(context, input_b.dims->size >= 2);
    TF_LITE_ENSURE_EQ(context, input_b.dims->data[input_b.dims->size - 1] % 2,
                      0);
  } else if (input_b.type == kTfLiteInt2) {
    TF_LITE_ENSURE(context, input_b.dims->size >= 2);
    TF_LITE_ENSURE_EQ(context, input_b.dims->data[input_b.dims->size - 1] % 4,
                      0);
  }

  TF_LITE_ENSURE(context, input_a.dims->size >= 2);
  TF_LITE_ENSURE(context, input_b.dims->size >= 2);
  TF_LITE_ENSURE(context, input_a.dims->size <= YNN_MAX_TENSOR_RANK);
  TF_LITE_ENSURE(context, input_b.dims->size <= YNN_MAX_TENSOR_RANK);

  return kTfLiteOk;
}

bool IsRuntimeBmm(const TfLiteRegistration* registration,
                  const TfLiteNode* node) {
  if (registration == nullptr) {
    return false;
  }
  if (registration->builtin_code == kTfLiteBuiltinCustom &&
      registration->custom_name != nullptr &&
      strcmp(registration->custom_name, "odml.runtime_bmm") == 0) {
    return true;
  }
  if (registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node != nullptr && node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    return composite_params->name != nullptr &&
           strcmp(composite_params->name, "odml.runtime_bmm") == 0;
  }
  return false;
}

bool IsRuntimeBmm(TfLiteContext* context, int node_index) {
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  if (context == nullptr ||
      context->GetNodeAndRegistration(context, node_index, &node,
                                      &registration) != kTfLiteOk) {
    return false;
  }
  return IsRuntimeBmm(registration, node);
}

TfLiteStatus IsRuntimeBatchedMatMulSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context) {
  TF_LITE_ENSURE(context, IsRuntimeBmm(registration, node));

  return IsBatchMatMulSupported(registration, node, context,
                                /*is_runtime_bmm=*/true);
}

TfLiteStatus IsFullyConnectedSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context) {
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& weights = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, tflite::NumElements(&input) > 0);
  TF_LITE_ENSURE(context, tflite::NumElements(&weights) > 0);

  bool has_bias = node->inputs->size == 3 && node->inputs->data[2] >= 0;

  TF_LITE_ENSURE(context, IsTensorSupported(input));
  TF_LITE_ENSURE(context,
                 IsTensorSupported(weights, /*allow_per_channel=*/true));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  if (has_bias) {
    const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
    TF_LITE_ENSURE(context,
                   IsTensorSupported(bias, /*allow_per_channel=*/true));
  }

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  if (input.type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, weights.type == kTfLiteInt8 ||
                                weights.type == kTfLiteInt4 ||
                                weights.type == kTfLiteUInt4 ||
                                weights.type == kTfLiteInt2);
    TF_LITE_ENSURE_EQ(context, output.type, kTfLiteInt8);
    if (has_bias) {
      const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
      TF_LITE_ENSURE_EQ(context, bias.type, kTfLiteInt32);
    }
  } else if (is_float_type(input.type)) {
    TF_LITE_ENSURE(context, is_float_type(weights.type) ||
                                weights.type == kTfLiteInt8 ||
                                weights.type == kTfLiteInt4 ||
                                weights.type == kTfLiteUInt4 ||
                                weights.type == kTfLiteInt2);
    TF_LITE_ENSURE(context, is_float_type(output.type));
    if (has_bias) {
      const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
      TF_LITE_ENSURE(context, is_float_type(bias.type));
    }
  } else {
    return kTfLiteError;
  }

  if (weights.type == kTfLiteInt4 || weights.type == kTfLiteUInt4) {
    TF_LITE_ENSURE_EQ(context, weights.dims->size, 2);
    TF_LITE_ENSURE_EQ(context, weights.dims->data[0] % 2, 0);
    TF_LITE_ENSURE_EQ(context, weights.dims->data[1] % 2, 0);
  } else if (weights.type == kTfLiteInt2) {
    TF_LITE_ENSURE_EQ(context, weights.dims->size, 2);
    TF_LITE_ENSURE_EQ(context, weights.dims->data[0] % 4, 0);
    TF_LITE_ENSURE_EQ(context, weights.dims->data[1] % 4, 0);
  }

  TF_LITE_ENSURE(context, input.dims->size >= 2);
  TF_LITE_ENSURE_EQ(context, weights.dims->size, 2);
  TF_LITE_ENSURE(context, input.dims->size <= YNN_MAX_TENSOR_RANK);

  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(context,
                 IsActivationSupported(params->activation, output.type));

  return kTfLiteOk;
}

TfLiteStatus DefineBatchMatMulNode(TfLiteContext* context,
                                   ynn_subgraph_t subgraph,
                                   TensorToValueIdMap& tensor_to_value_id,
                                   const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_a_tensor_index = node.inputs[0];
  int input_b_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_a_tensor = context->tensors[input_a_tensor_index];
  const TfLiteTensor& input_b_tensor = context->tensors[input_b_tensor_index];

  uint32_t input_a_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_a_tensor_index);
  uint32_t input_b_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_b_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_a_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, input_b_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteBatchMatMulParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int rank_a = input_a_tensor.dims->size;
  int rank_b = input_b_tensor.dims->size;

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  TF_LITE_ENSURE_STATUS(DefineMatMul(
      context, subgraph, rank_a, rank_b, input_a_val_id, input_b_val_id,
      YNN_INVALID_VALUE_ID, params->adj_x, params->adj_y,
      /*mutual_broadcast=*/true, input_a_tensor, input_b_tensor, output_tensor,
      &output_val_id));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

namespace {

flexbuffers::Map GetFlexBufferMap(const TfLiteRegistration* reg,
                                  const TfLiteNode* node) {
  if (node == nullptr) {
    return flexbuffers::Map::EmptyMap();
  }
  if (reg != nullptr && reg->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    if (composite_params->attributes != nullptr &&
        composite_params->attributes_size > 0) {
      return flexbuffers::GetRoot(composite_params->attributes,
                                  composite_params->attributes_size)
          .AsMap();
    }
  }
  if (node->custom_initial_data != nullptr &&
      node->custom_initial_data_size > 0) {
    return flexbuffers::GetRoot(
               reinterpret_cast<const uint8_t*>(node->custom_initial_data),
               node->custom_initial_data_size)
        .AsMap();
  }
  return flexbuffers::Map::EmptyMap();
}

// Find a dummy input we can use for a particular runtime_bmm op. Often, many
// runtime_bmm ops use the same params tensor, which can share a dummy input.
TfLiteStatus GetOrCreateDummyInput(TfLiteContext* context,
                                   ynn_subgraph_t subgraph,
                                   uint32_t& next_external_id,
                                   std::vector<DummyInputInfo>& dummy_inputs,
                                   int param_tensor_index, int seq_axis,
                                   size_t rank, const size_t* full_dims,
                                   ynn_type type, uint32_t* dummy_val_id_out) {
  for (const auto& dummy : dummy_inputs) {
    if (dummy.param_tensor_index == param_tensor_index &&
        dummy.seq_axis == seq_axis && dummy.rank == rank) {
      bool dims_match = true;
      for (size_t i = 0; i < rank; ++i) {
        if (dummy.full_dims[i] != full_dims[i]) {
          dims_match = false;
          break;
        }
      }
      if (dims_match) {
        *dummy_val_id_out = dummy.dummy_val_id;
        return kTfLiteOk;
      }
    }
  }

  uint32_t dummy_val_id = next_external_id++;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
      subgraph, type, rank, /*dims=*/nullptr, /*data=*/nullptr,
      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dummy_val_id));

  DummyInputInfo dummy_info;
  dummy_info.param_tensor_index = param_tensor_index;
  dummy_info.dummy_val_id = dummy_val_id;
  dummy_info.seq_axis = seq_axis;
  dummy_info.rank = rank;
  std::copy_n(full_dims, rank, dummy_info.full_dims);
  dummy_inputs.push_back(dummy_info);

  *dummy_val_id_out = dummy_val_id;
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus DefineRuntimeBatchedMatMulNode(
    TfLiteContext* context, ynn_subgraph_t subgraph,
    TensorToValueIdMap& tensor_to_value_id, uint32_t& next_external_id,
    std::vector<DummyInputInfo>& dummy_inputs, const NodeInfo& node) {
  TF_LITE_ENSURE(context, node.inputs.size() >= 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_a_tensor_index = node.inputs[0];
  int input_b_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_a_tensor = context->tensors[input_a_tensor_index];
  const TfLiteTensor& input_b_tensor = context->tensors[input_b_tensor_index];
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  uint32_t input_a_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_a_tensor_index);
  uint32_t input_b_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_b_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_a_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, input_b_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  int rank_a = input_a_tensor.dims->size;
  int rank_b = input_b_tensor.dims->size;

  const bool has_param_tensor = (node.inputs.size() >= 3);

  bool adj_x = false;
  bool adj_y = false;
  bool is_src = false;
  TfLiteNode* tflite_node = nullptr;
  TfLiteRegistration* reg = nullptr;
  if (context->GetNodeAndRegistration(context, node.node_index, &tflite_node,
                                      &reg) == kTfLiteOk &&
      tflite_node != nullptr) {
    if (reg->builtin_code == kTfLiteBuiltinBatchMatmul) {
      if (tflite_node->builtin_data != nullptr) {
        const auto* params = static_cast<const TfLiteBatchMatMulParams*>(
            tflite_node->builtin_data);
        if (params) {
          adj_x = params->adj_x;
          adj_y = params->adj_y;
        }
      }
    } else {
      const flexbuffers::Map flexbuffer_map =
          GetFlexBufferMap(reg, tflite_node);
      if (!flexbuffer_map["is_src"].IsNull()) {
        is_src = flexbuffer_map["is_src"].AsBool();
      }
    }
  }

  if (!is_src && rank_a >= 2 && rank_b >= 2) {
    // Sometimes the odml.runtime_bmm op does not specify `is_src`, but it
    // should be, we need to look at the shape to determine this...?
    if (input_a_tensor.dims->data[rank_a - 1] ==
            input_b_tensor.dims->data[rank_b - 2] &&
        input_a_tensor.dims->data[rank_a - 1] !=
            input_b_tensor.dims->data[rank_b - 1]) {
      is_src = true;
    }
  }

  uint32_t current_a_val_id = input_a_val_id;
  uint32_t current_b_val_id = input_b_val_id;

  if (has_param_tensor) {
    int param_tensor_index = node.inputs[2];

    if (!is_src) {
      // is_src = false (Q * K^T): Compute MatMul and slice/pad the output along
      // sequence axis (rank_a - 1) to active_tokens while keeping output shape.
      uint32_t full_matmul_val_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(DefineMatMul(
          context, subgraph, rank_a, rank_b, current_a_val_id, current_b_val_id,
          YNN_INVALID_VALUE_ID, adj_x, /*adj_y=*/true,
          /*mutual_broadcast=*/true, input_a_tensor, input_b_tensor,
          output_tensor, &full_matmul_val_id));

      int seq_axis_out = rank_a - 1;
      size_t full_dims_out[YNN_MAX_TENSOR_RANK];
      for (int i = 0; i < rank_a; ++i) {
        full_dims_out[i] = output_tensor.dims->data[i];
      }
      uint32_t dummy_val_id_out = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
          context, subgraph, next_external_id, dummy_inputs, param_tensor_index,
          seq_axis_out, rank_a, full_dims_out, GetYnnType(output_tensor.type),
          &dummy_val_id_out));

      int32_t slice_axes_out[1] = {seq_axis_out};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, /*num_axes=*/1, slice_axes_out, full_matmul_val_id,
          dummy_val_id_out, &output_val_id, YNN_NODE_FLAG_KEEP_SHAPE));
    } else {
      // is_src = true (P * V):
      // Slice input_a (P) along seq_axis_a (rank_a - 1)
      // Slice input_b (V) along seq_axis_b (rank_b - 1)
      int seq_axis_a = rank_a - 1;
      size_t full_dims_a[YNN_MAX_TENSOR_RANK];
      for (int i = 0; i < rank_a; ++i) {
        full_dims_a[i] = input_a_tensor.dims->data[i];
      }
      full_dims_a[seq_axis_a] =
          std::min<size_t>(input_a_tensor.dims->data[seq_axis_a],
                           input_b_tensor.dims->data[rank_b - 1]);

      uint32_t dummy_val_id_a = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
          context, subgraph, next_external_id, dummy_inputs, param_tensor_index,
          seq_axis_a, rank_a, full_dims_a, GetYnnType(input_a_tensor.type),
          &dummy_val_id_a));

      uint32_t sliced_a_val_id = YNN_INVALID_VALUE_ID;
      int32_t slice_axes_a[1] = {seq_axis_a};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, /*num_axes=*/1, slice_axes_a, current_a_val_id,
          dummy_val_id_a, &sliced_a_val_id, /*flags=*/0));
      current_a_val_id = sliced_a_val_id;

      int seq_axis_b = rank_b - 1;
      size_t full_dims_b[YNN_MAX_TENSOR_RANK];
      for (int i = 0; i < rank_b; ++i) {
        full_dims_b[i] = input_b_tensor.dims->data[i];
      }
      uint32_t dummy_val_id_b = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
          context, subgraph, next_external_id, dummy_inputs, param_tensor_index,
          seq_axis_b, rank_b, full_dims_b, GetYnnType(input_b_tensor.type),
          &dummy_val_id_b));

      uint32_t sliced_b_val_id = YNN_INVALID_VALUE_ID;
      int32_t slice_axes_b[1] = {seq_axis_b};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, /*num_axes=*/1, slice_axes_b, current_b_val_id,
          dummy_val_id_b, &sliced_b_val_id, /*flags=*/0));
      current_b_val_id = sliced_b_val_id;

      TF_LITE_ENSURE_STATUS(DefineMatMul(
          context, subgraph, rank_a, rank_b, current_a_val_id, current_b_val_id,
          YNN_INVALID_VALUE_ID, /*adj_x=*/false, /*adj_y=*/true,
          /*mutual_broadcast=*/true, input_a_tensor, input_b_tensor,
          output_tensor, &output_val_id));
    }
  } else {
    TF_LITE_ENSURE_STATUS(
        DefineMatMul(context, subgraph, rank_a, rank_b, current_a_val_id,
                     current_b_val_id, YNN_INVALID_VALUE_ID, adj_x, adj_y,
                     /*mutual_broadcast=*/true, input_a_tensor, input_b_tensor,
                     output_tensor, &output_val_id));
  }

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineFullyConnectedNode(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      TensorToValueIdMap& tensor_to_value_id,
                                      const NodeInfo& node) {
  const int input_tensor_index = node.inputs[0];
  const int weights_tensor_index = node.inputs[1];
  const int bias_tensor_index = (node.inputs.size() == 3) ? node.inputs[2] : -1;
  const int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& weights_tensor = context->tensors[weights_tensor_index];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t weights_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, weights_tensor_index);
  uint32_t bias_val_id = YNN_INVALID_VALUE_ID;
  if (bias_tensor_index != -1) {
    bias_val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                     bias_tensor_index);
  }
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, weights_val_id != YNN_INVALID_VALUE_ID);
  if (bias_tensor_index != -1) {
    TF_LITE_ENSURE(context, bias_val_id != YNN_INVALID_VALUE_ID);
  }
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int rank_a = input_tensor.dims->size;
  int rank_b = weights_tensor.dims->size;

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  uint32_t reshaped_input_val_id = input_val_id;
  if (!params->keep_num_dims) {
    size_t input_channels = weights_tensor.dims->data[1];
    size_t new_dims[2] = {0, input_channels};
    reshaped_input_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
        subgraph, 2, new_dims, input_val_id, &reshaped_input_val_id, 0));
    rank_a = 2;
  }

  // If activation is present, we must create a temporary tensor for the
  // MatMul output, and then apply activation to it, writing to output_val_id.
  uint32_t matmul_output_id = params->activation != kTfLiteActNone
                                  ? YNN_INVALID_VALUE_ID
                                  : output_val_id;

  TF_LITE_ENSURE_STATUS(
      DefineMatMul(context, subgraph, rank_a, rank_b, reshaped_input_val_id,
                   weights_val_id, bias_val_id, /*adj_x=*/false, /*adj_y=*/true,
                   /*mutual_broadcast=*/false, input_tensor, weights_tensor,
                   output_tensor, &matmul_output_id));

  if (params->activation != kTfLiteActNone) {
    TF_LITE_ENSURE_STATUS(ApplyActivation(
        context, subgraph, params->activation, matmul_output_id, output_val_id,
        output_tensor_index, GetYnnType(output_tensor.type)));
  }

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus IsConvSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& filter = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, IsTensorSupported(input));
  TF_LITE_ENSURE(context,
                 IsTensorSupported(filter, /*allow_per_channel=*/true));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  TF_LITE_ENSURE(context, IsTensorSupported(bias, /*allow_per_channel=*/true));

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  bool is_quantized = input.type == kTfLiteInt8 || input.type == kTfLiteUInt8;

  if (is_quantized) {
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
    TF_LITE_ENSURE_EQ(context, filter.type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, bias.type, kTfLiteInt32);
  } else if (is_float_type(input.type)) {
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
    TF_LITE_ENSURE_EQ(context, filter.type, input.type);
    TF_LITE_ENSURE_EQ(context, bias.type, input.type);
  } else {
    return kTfLiteError;
  }

  TF_LITE_ENSURE_EQ(context, input.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, bias.dims->size, 1);
  if (output.dims->size > 0) {
    TF_LITE_ENSURE_EQ(context, output.dims->size, 4);
  }

  const auto* params = static_cast<const TfLiteConvParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);
  TF_LITE_ENSURE(context, params->dilation_height_factor > 0);
  TF_LITE_ENSURE(context, params->dilation_width_factor > 0);

  TF_LITE_ENSURE(context,
                 IsActivationSupported(params->activation, output.type));

  int output_channels = filter.dims->data[0];
  int input_channels_per_group = filter.dims->data[3];

  if (output.dims->size > 0) {
    TF_LITE_ENSURE_EQ(context, output.dims->data[3], output_channels);
  }
  TF_LITE_ENSURE_EQ(context, bias.dims->data[0], output_channels);

  int input_channels = input.dims->data[3];
  TF_LITE_ENSURE_EQ(context, input_channels % input_channels_per_group, 0);

  return kTfLiteOk;
}

TfLiteStatus IsDepthwiseConvSupported(const TfLiteRegistration* registration,
                                      const TfLiteNode* node,
                                      TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& filter = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, IsTensorSupported(input));
  TF_LITE_ENSURE(context,
                 IsTensorSupported(filter, /*allow_per_channel=*/true));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  TF_LITE_ENSURE(context, IsTensorSupported(bias));
  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  bool is_quantized = input.type == kTfLiteInt8 || input.type == kTfLiteUInt8;

  if (is_quantized) {
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
    TF_LITE_ENSURE_EQ(context, filter.type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, bias.type, kTfLiteInt32);
  } else if (is_float_type(input.type)) {
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
    TF_LITE_ENSURE_EQ(context, filter.type, input.type);
    TF_LITE_ENSURE_EQ(context, bias.type, input.type);
  } else {
    return kTfLiteError;
  }

  TF_LITE_ENSURE_EQ(context, input.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, bias.dims->size, 1);
  if (output.dims->size > 0) {
    TF_LITE_ENSURE_EQ(context, output.dims->size, 4);
  }

  const auto* params =
      static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);
  TF_LITE_ENSURE(context, params->dilation_height_factor > 0);
  TF_LITE_ENSURE(context, params->dilation_width_factor > 0);

  TF_LITE_ENSURE(context,
                 IsActivationSupported(params->activation, output.type));

  TF_LITE_ENSURE_EQ(context, filter.dims->data[0], 1);
  int filter_channels = filter.dims->data[3];

  int input_channels = input.dims->data[3];
  TF_LITE_ENSURE(context, input_channels > 0);
  TF_LITE_ENSURE_EQ(context, filter_channels % input_channels, 0);
  int depth_multiplier = filter_channels / input_channels;

  if (output.dims->size > 0) {
    TF_LITE_ENSURE_EQ(context, output.dims->data[3],
                      input_channels * depth_multiplier);
  }
  TF_LITE_ENSURE_EQ(context, bias.dims->data[0],
                    input_channels * depth_multiplier);

  if (params->depth_multiplier > 0) {
    TF_LITE_ENSURE_EQ(context, params->depth_multiplier, depth_multiplier);
  }

  return kTfLiteOk;
}

namespace {

TfLiteStatus DefineConv(TfLiteContext* context, ynn_subgraph_t subgraph,
                        uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
                        uint32_t output_id, const TfLiteTensor& input_tensor,
                        const TfLiteTensor& filter_tensor,
                        const TfLiteTensor& output_tensor, int stride_height,
                        int stride_width, int dilation_height,
                        int dilation_width, TfLitePadding padding,
                        TfLiteFusedActivation activation, size_t groups,
                        size_t group_input_channels,
                        size_t group_output_channels,
                        size_t output_tensor_index) {
  bool is_quantized = IsQuantized(input_tensor);
  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t out_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t out_zp_id = YNN_INVALID_VALUE_ID;

  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, input_tensor, &a_scale_id, &a_zp_id));
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, filter_tensor, &b_scale_id, &b_zp_id));
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, output_tensor, &out_scale_id, &out_zp_id));
  int kernel_height = filter_tensor.dims->data[1];
  int kernel_width = filter_tensor.dims->data[2];

  bool is_per_channel = false;
  if (is_quantized) {
    const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
        filter_tensor.quantization.params);
    is_per_channel =
        quant_params && quant_params->scale && quant_params->scale->size > 1;
  }

  float padding_value = 0.0f;
  if (is_quantized) {
    const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
        input_tensor.quantization.params);
    if (quant_params && quant_params->zero_point) {
      padding_value = quant_params->zero_point->data[0];
    } else {
      padding_value = input_tensor.params.zero_point;
    }
  }

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineYnnStencil(
      context, subgraph, input_tensor, input_id, kernel_height, kernel_width,
      stride_height, stride_width, dilation_height, dilation_width, padding,
      padding_value, &stencil_id));

  uint32_t current_input_id = stencil_id;
  uint32_t current_filter_id = filter_id;
  uint32_t current_bias_id = bias_id;
  uint32_t current_b_scale_id = b_scale_id;
  uint32_t current_b_zp_id = b_zp_id;

  if (groups != 1) {
    // Split input: [n, h, w, kh, kw, ci] -> [n, h, w, kh, kw, g, 1, ci/g]
    uint32_t split_input_id = YNN_INVALID_VALUE_ID;
    const size_t input_split[] = {groups, 1, group_input_channels};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
        subgraph, 5, 3, input_split, current_input_id, &split_input_id, 0));
    current_input_id = split_input_id;

    // Split filter: [co, kh, kw, ci/g] -> [g, co/g, kh, kw, ci/g]
    uint32_t split_filter_id = YNN_INVALID_VALUE_ID;
    const size_t filter_split[] = {groups, group_output_channels};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
        subgraph, 0, 2, filter_split, current_filter_id, &split_filter_id, 0));
    current_filter_id = split_filter_id;

    // Split bias if present: [co] -> [g, 1, co/g]
    if (current_bias_id != YNN_INVALID_VALUE_ID) {
      uint32_t split_bias_id = YNN_INVALID_VALUE_ID;
      const size_t bias_split[] = {groups, 1, group_output_channels};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
          subgraph, 0, 3, bias_split, current_bias_id, &split_bias_id, 0));
      current_bias_id = split_bias_id;
    }

    if (is_quantized && is_per_channel) {
      if (current_b_scale_id != YNN_INVALID_VALUE_ID) {
        uint32_t split_scale_id = YNN_INVALID_VALUE_ID;
        const size_t scale_split[] = {groups, group_output_channels};
        TF_LITE_ENSURE_YNN_STATUS(
            ynn_define_split_dim(subgraph, 0, 2, scale_split,
                                 current_b_scale_id, &split_scale_id, 0));
        current_b_scale_id = split_scale_id;
      }
      if (current_b_zp_id != YNN_INVALID_VALUE_ID) {
        uint32_t split_zp_id = YNN_INVALID_VALUE_ID;
        const size_t zp_split[] = {groups, group_output_channels};
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
            subgraph, 0, 2, zp_split, current_b_zp_id, &split_zp_id, 0));
        current_b_zp_id = split_zp_id;
      }
    }

    // Transpose input: [n, h, w, kh, kw, g, 1, ci/g] -> [n, h, w, g, 1, kh, kw,
    // ci/g]
    uint32_t transposed_input_id = YNN_INVALID_VALUE_ID;
    const int32_t input_perm[] = {0, 1, 2, 5, 6, 3, 4, 7};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 8, input_perm, current_input_id, &transposed_input_id, 0));
    current_input_id = transposed_input_id;
  }

  // Transpose filter:
  // If groups == 1: [co, kh, kw, ci] -> [kh, kw, ci, co]
  // If groups > 1: [g, co/g, kh, kw, ci/g] -> [g, kh, kw, ci/g, co/g]
  uint32_t transposed_filter_id = YNN_INVALID_VALUE_ID;
  if (groups == 1) {
    int32_t swap_co_ci[4] = {1, 2, 3, 0};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, swap_co_ci, current_filter_id, &transposed_filter_id, 0));
  } else {
    int32_t swap_co_ci[5] = {0, 2, 3, 4, 1};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 5, swap_co_ci, current_filter_id, &transposed_filter_id, 0));
  }

  uint32_t conv_output_id = output_id;
  if (activation != kTfLiteActNone) {
    conv_output_id = YNN_INVALID_VALUE_ID;
  }

  uint32_t dot_output_id =
      (groups == 1) ? conv_output_id : YNN_INVALID_VALUE_ID;

  if (is_quantized) {
    int32_t a_reduce_axes[3];
    int32_t b_reduce_axes[3];
    std::iota(a_reduce_axes, a_reduce_axes + 3, groups == 1 ? 3 : 5);
    std::iota(b_reduce_axes, b_reduce_axes + 3, groups == 1 ? 0 : 1);

    TF_LITE_ENSURE_STATUS(DefineQuantizedDot(
        context, subgraph,
        /*rank_a=*/(groups == 1) ? 6 : 8,
        /*rank_b=*/(groups == 1) ? 4 : 5, a_reduce_axes, b_reduce_axes,
        current_input_id, a_scale_id, a_zp_id, transposed_filter_id,
        current_b_scale_id, current_b_zp_id, current_bias_id,
        (groups == 1) ? 1 : 3, out_scale_id, out_zp_id, is_per_channel,
        /*is_conv=*/true, GetYnnType(output_tensor.type), &dot_output_id));
  } else {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dot(subgraph, 3, current_input_id, transposed_filter_id,
                       current_bias_id, &dot_output_id, 0));
  }

  if (groups != 1) {
    // Fuse [n, h, w, g, 1, co/g] -> [n, h, w, co]
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_fuse_dim(subgraph, 3, 3, dot_output_id, &conv_output_id, 0));
  } else {
    conv_output_id = dot_output_id;
  }

  if (activation != kTfLiteActNone) {
    TF_LITE_ENSURE_STATUS(ApplyActivation(
        context, subgraph, activation, conv_output_id, output_id,
        output_tensor_index, GetYnnType(output_tensor.type)));
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteStatus DefineConvNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            const NodeInfo& node) {
  int input_tensor_index = node.inputs[0];
  int filter_tensor_index = node.inputs[1];
  int bias_tensor_index = node.inputs[2];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& filter_tensor = context->tensors[filter_tensor_index];
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  uint32_t input_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         input_tensor_index);
  uint32_t filter_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          filter_tensor_index);
  uint32_t bias_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                        bias_tensor_index);
  uint32_t output_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          output_tensor_index);

  TF_LITE_ENSURE(context, input_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, filter_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, bias_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteConvParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int output_channels = filter_tensor.dims->data[0];
  int input_channels_per_group = filter_tensor.dims->data[3];
  int input_channels = input_tensor.dims->data[3];
  int groups = input_channels / input_channels_per_group;
  int group_input_channels = input_channels_per_group;
  int group_output_channels = output_channels / groups;

  TF_LITE_ENSURE_STATUS(DefineConv(
      context, subgraph, input_id, filter_id, bias_id, output_id, input_tensor,
      filter_tensor, output_tensor, params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor,
      params->padding, params->activation, groups, group_input_channels,
      group_output_channels, output_tensor_index));

  tensor_to_value_id[output_tensor_index] = output_id;
  return kTfLiteOk;
}

TfLiteStatus DefineDepthwiseConvNode(TfLiteContext* context,
                                     ynn_subgraph_t subgraph,
                                     TensorToValueIdMap& tensor_to_value_id,
                                     const NodeInfo& node) {
  int input_tensor_index = node.inputs[0];
  int filter_tensor_index = node.inputs[1];
  int bias_tensor_index = node.inputs[2];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& filter_tensor = context->tensors[filter_tensor_index];
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  uint32_t input_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         input_tensor_index);
  uint32_t filter_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          filter_tensor_index);
  uint32_t bias_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                        bias_tensor_index);
  uint32_t output_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          output_tensor_index);

  TF_LITE_ENSURE(context, input_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, filter_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, bias_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteDepthwiseConvParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int filter_channels = filter_tensor.dims->data[3];
  int input_channels = input_tensor.dims->data[3];
  int depth_multiplier = filter_channels / input_channels;

  // Transpose filter: [1, kh, kw, ci * dm] -> [ci * dm, kh, kw, 1]
  const int32_t swap_dims[4] = {3, 1, 2, 0};
  uint32_t transposed_filter_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 4, swap_dims, filter_id, &transposed_filter_id, 0));

  int groups = input_channels;
  int group_input_channels = 1;
  int group_output_channels = depth_multiplier;
  TF_LITE_ENSURE_STATUS(DefineConv(
      context, subgraph, input_id, transposed_filter_id, bias_id, output_id,
      input_tensor, filter_tensor, output_tensor, params->stride_height,
      params->stride_width, params->dilation_height_factor,
      params->dilation_width_factor, params->padding, params->activation,
      groups, group_input_channels, group_output_channels,
      output_tensor_index));

  tensor_to_value_id[output_tensor_index] = output_id;
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
