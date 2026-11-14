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
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {

namespace {

bool IsAllZero(const TfLiteTensor& tensor) {
  const int count = tflite::NumElements(&tensor);
  switch (tensor.type) {
    case kTfLiteFloat32:
      return std::all_of(tensor.data.f, tensor.data.f + count,
                         [](float v) { return v == 0.0f; });
    case kTfLiteInt32:
      return std::all_of(tensor.data.i32, tensor.data.i32 + count,
                         [](int32_t v) { return v == 0; });
    case kTfLiteInt8:
      return std::all_of(tensor.data.int8, tensor.data.int8 + count,
                         [](int8_t v) { return v == 0; });
    case kTfLiteUInt8:
      return std::all_of(tensor.data.uint8, tensor.data.uint8 + count,
                         [](uint8_t v) { return v == 0; });
    case kTfLiteFloat16:
    case kTfLiteBFloat16: {
      const auto* data = reinterpret_cast<const uint16_t*>(tensor.data.raw);
      return std::all_of(data, data + count,
                         [](uint16_t bits) { return (bits & 0x7fff) == 0; });
    }
    default:
      return false;
  }
}

struct MatMulWeightQuantization {
  // Tensor indices for tensor-backed quantization (Blockwise, MultiAxis,
  // BlockwiseV2); -1 for Affine/legacy or unquantized.
  int scale_tensor_index = -1;
  int zp_tensor_index = -1;
  // > 0 when quantization varies along K and needs ynn::define_blockwise_dot.
  int block_size = 0;
  // True when scale/zp tensors are rank-N in weights' dimension order
  // (per-channel Affine or BlockwiseV2), rather than scalar or
  // [..., N, num_blocks] (Blockwise v1 / MultiAxis).
  bool has_weight_layout = false;
};

// Validates and parses quantization parameters for matmul weights reduced along
// `k_axis`. When `options` is non-null, enforces
// `enable_blockwise_quantization` if quantization varies along `k_axis`.
TfLiteStatus GetMatMulWeightQuantization(
    TfLiteContext* context, const TfLiteTensor& weights, int k_axis,
    const TfLiteYNNPackDelegateOptions* options,
    MatMulWeightQuantization* info) {
  *info = {};
  if (!IsQuantized(weights)) return kTfLiteOk;
  TF_LITE_ENSURE(context, weights.dims != nullptr && weights.dims->size >= 2);
  const int rank = weights.dims->size;
  const int n_axis = (k_axis == rank - 1) ? rank - 2 : rank - 1;

  auto validate_param_tensor = [&](int index) -> TfLiteStatus {
    TF_LITE_ENSURE(context, index >= 0 && index < context->tensors_size);
    const TfLiteTensor& t = context->tensors[index];
    TF_LITE_ENSURE(context, IsConstant(t) && t.data.raw != nullptr &&
                                t.dims != nullptr && IsTensorTypeSupported(t) &&
                                t.type != kTfLiteFloat64 &&
                                YnnTypeElementCount(GetYnnType(t.type)) == 1);
    return kTfLiteOk;
  };

  switch (weights.quantization.type) {
    case kTfLiteNoQuantization:
    case kTfLiteAffineQuantization: {
      TF_LITE_ENSURE(context, IsSupportedQuantization(
                                  weights, /*allow_per_channel=*/true));
      if (weights.quantization.type == kTfLiteAffineQuantization) {
        const auto* params = static_cast<const TfLiteAffineQuantization*>(
            weights.quantization.params);
        if (params->scale->size > 1) {
          TF_LITE_ENSURE_EQ(context, params->quantized_dimension, n_axis);
          TF_LITE_ENSURE_EQ(context, params->scale->size,
                            weights.dims->data[n_axis]);
          info->has_weight_layout = true;
        }
      }
      return kTfLiteOk;
    }
    case kTfLiteBlockwiseQuantization:
    case kTfLiteMultiAxisQuantization: {
      int scale_idx = -1;
      int zp_idx = -1;
      int blocksize = 0;
      if (weights.quantization.type == kTfLiteBlockwiseQuantization) {
        const auto* params = static_cast<const TfLiteBlockwiseQuantization*>(
            weights.quantization.params);
        TF_LITE_ENSURE(context, params != nullptr && params->blocksize > 0);
        scale_idx = params->scale;
        zp_idx = params->zero_point;
        blocksize = params->blocksize;
      } else {
        const auto* params = static_cast<const TfLiteMultiAxisQuantization*>(
            weights.quantization.params);
        TF_LITE_ENSURE(context, params != nullptr && params->blocksize >= 0);
        scale_idx = params->scales;
        zp_idx = params->zero_points;
        blocksize = params->blocksize;
      }
      TF_LITE_ENSURE_STATUS(validate_param_tensor(scale_idx));
      if (zp_idx >= 0) {
        TF_LITE_ENSURE_STATUS(validate_param_tensor(zp_idx));
      }
      if (blocksize > 0) {
        if (options) {
          TF_LITE_ENSURE(context, options->enable_blockwise_quantization);
        }
        TF_LITE_ENSURE_EQ(context, weights.dims->data[k_axis] % blocksize, 0);
        info->block_size = blocksize;
      }
      info->scale_tensor_index = scale_idx;
      if (zp_idx >= 0 && !IsAllZero(context->tensors[zp_idx])) {
        info->zp_tensor_index = zp_idx;
      }
      return kTfLiteOk;
    }
    case kTfLiteBlockwiseQuantizationV2: {
      const auto* params = static_cast<const TfLiteBlockwiseQuantizationV2*>(
          weights.quantization.params);
      TF_LITE_ENSURE(context,
                     params != nullptr && params->block_shape != nullptr);
      TF_LITE_ENSURE_EQ(context, params->block_shape->size, rank);
      TF_LITE_ENSURE_STATUS(validate_param_tensor(params->scale));
      const TfLiteTensor& scale_tensor = context->tensors[params->scale];
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->size, rank);
      const TfLiteTensor* zp_tensor = nullptr;
      if (params->zero_point >= 0) {
        TF_LITE_ENSURE_STATUS(validate_param_tensor(params->zero_point));
        zp_tensor = &context->tensors[params->zero_point];
        TF_LITE_ENSURE_EQ(context, zp_tensor->dims->size, rank);
      }
      const bool has_zp = zp_tensor != nullptr && !IsAllZero(*zp_tensor);
      for (int d = 0; d < rank; ++d) {
        const int dim = weights.dims->data[d];
        const int block = params->block_shape->data[d];
        TF_LITE_ENSURE(context, block > 0);
        const int num_blocks = (dim == 0) ? 0 : 1 + (dim - 1) / block;
        const int scale_extent = scale_tensor.dims->data[d];
        TF_LITE_ENSURE(context,
                       scale_extent == 1 || scale_extent == num_blocks);
        int zp_extent = 1;
        if (zp_tensor != nullptr) {
          zp_extent = zp_tensor->dims->data[d];
          TF_LITE_ENSURE(context, zp_extent == 1 || zp_extent == num_blocks);
        }
        const bool varies = scale_extent > 1 || (has_zp && zp_extent > 1);
        if (!varies) continue;
        if (d == k_axis) {
          if (options) {
            TF_LITE_ENSURE(context, options->enable_blockwise_quantization);
          }
          TF_LITE_ENSURE(context, block < dim && dim % block == 0);
          info->block_size = block;
        } else {
          TF_LITE_ENSURE_EQ(context, block, 1);
        }
      }
      info->scale_tensor_index = params->scale;
      info->zp_tensor_index = has_zp ? params->zero_point : -1;
      info->has_weight_layout = true;
      return kTfLiteOk;
    }
    default:
      return kTfLiteError;
  }
}

TfLiteStatus DefineQuantizedDot(TfLiteContext* context, ynn_subgraph_t subgraph,
                                int num_k_dims, uint32_t a_id,
                                uint32_t a_scale_id, uint32_t a_zp_id,
                                uint32_t b_id, uint32_t b_scale_id,
                                uint32_t b_zp_id, uint32_t bias_id,
                                uint32_t out_scale_id, uint32_t out_zp_id,
                                ynn_type output_ynn_type, uint32_t* output_id,
                                uint32_t dot_flags,
                                uint32_t float_bias_id = YNN_INVALID_VALUE_ID) {
  // We assume a_id and b_id are quantized (int8 or uint8).
  // Accumulator will be int32.

  bool is_dynamically_quantized = (out_scale_id == YNN_INVALID_VALUE_ID);

  // Compute zero point and scale of the dot product.
  uint32_t dot_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_dot_quantization(
      subgraph, num_k_dims, a_id, a_zp_id, a_scale_id, b_id, b_zp_id,
      b_scale_id, dot_zp_id, dot_scale_id));

  // Define the dot product and subtract zero-point contribution afterward so
  // floating-point zero points are not truncated to the integer accumulator.
  uint32_t accum_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, num_k_dims, a_id, b_id,
                                           bias_id, &accum_id, dot_flags));

  if (dot_zp_id != YNN_INVALID_VALUE_ID) {
    uint32_t adjusted_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_subtract, accum_id, dot_zp_id, &adjusted_id, 0));
    accum_id = adjusted_id;
  }

  if (is_dynamically_quantized && float_bias_id == YNN_INVALID_VALUE_ID) {
    // Dequantize accumulator directly to output.
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                              dot_scale_id, output_ynn_type, output_id, 0));
    return kTfLiteOk;
  }

  // Dequantize accumulator to float.
  uint32_t float_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                            dot_scale_id, ynn_type_fp32, &float_id, 0));

  if (float_bias_id != YNN_INVALID_VALUE_ID) {
    const bool direct_fp32_out =
        is_dynamically_quantized && output_ynn_type == ynn_type_fp32;
    uint32_t biased_id = direct_fp32_out ? *output_id : YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_add, float_id, float_bias_id, &biased_id, 0));
    if (direct_fp32_out) {
      *output_id = biased_id;
      return kTfLiteOk;
    }
    float_id = biased_id;
  }

  if (is_dynamically_quantized) {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_convert(subgraph, float_id, output_ynn_type, output_id, 0));
  } else {
    // Quantize back to output.
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(subgraph, float_id,
                                                  output_ynn_type, out_zp_id,
                                                  out_scale_id, output_id, 0));
  }

  return kTfLiteOk;
}

TfLiteStatus DefineMatMul(TfLiteContext* context, ynn_subgraph_t subgraph,
                          TensorToValueIdMap& tensor_to_value_id, int rank_a,
                          int rank_b, uint32_t a_id, uint32_t b_id,
                          uint32_t bias_id, bool adj_x, bool adj_y,
                          bool mutual_broadcast,
                          bool asymmetric_quantize_inputs,
                          const TfLiteTensor& input_a_tensor,
                          const TfLiteTensor& input_b_tensor,
                          const TfLiteTensor& output_tensor,
                          uint32_t* output_id, int bias_tensor_index = -1) {
  MatMulWeightQuantization weight_quant;
  const int k_axis = adj_y ? rank_b - 1 : rank_b - 2;
  TF_LITE_ENSURE_STATUS(GetMatMulWeightQuantization(
      context, input_b_tensor, k_axis, /*options=*/nullptr, &weight_quant));
  const bool is_blockwise = weight_quant.block_size > 0;

  bool is_input_a_quantized = IsQuantized(input_a_tensor);
  bool is_input_b_quantized = IsQuantized(input_b_tensor);
  bool is_output_quantized = IsQuantized(output_tensor);

  bool is_quantized = is_input_a_quantized && is_output_quantized;
  bool is_dynamically_quantized =
      !is_input_a_quantized && is_input_b_quantized && !is_output_quantized;

  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t out_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t out_zp_id = YNN_INVALID_VALUE_ID;

  if (weight_quant.scale_tensor_index >= 0) {
    b_scale_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                    weight_quant.scale_tensor_index);
    TF_LITE_ENSURE(context, b_scale_id != YNN_INVALID_VALUE_ID);
    if (weight_quant.zp_tensor_index >= 0) {
      b_zp_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                   weight_quant.zp_tensor_index);
      TF_LITE_ENSURE(context, b_zp_id != YNN_INVALID_VALUE_ID);
    }
  } else {
    TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
        context, subgraph, input_b_tensor, &b_scale_id, &b_zp_id));
  }
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
    uint32_t* a_zp_ptr = asymmetric_quantize_inputs ? &a_zp_id : nullptr;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dynamic_quantization(
        subgraph, min_max_id, ynn_type_int8, a_zp_ptr, &a_scale_id, 0));

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

  auto transpose = [&](uint32_t& val_id) -> TfLiteStatus {
    if (val_id == YNN_INVALID_VALUE_ID) return kTfLiteOk;
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;
    const int32_t perm[2] = {-1, -2};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 2, perm, val_id, &transposed_id, YNN_NODE_FLAG_KEEP_DIMS));
    val_id = transposed_id;
    return kTfLiteOk;
  };

  if (adj_x) {
    TF_LITE_ENSURE_STATUS(transpose(current_a_id));
    if (is_dynamically_quantized) {
      TF_LITE_ENSURE_STATUS(transpose(a_zp_id));
      TF_LITE_ENSURE_STATUS(transpose(a_scale_id));
    }
  }
  if (adj_y) {
    TF_LITE_ENSURE_STATUS(transpose(current_b_id));
  }
  // Per-channel Affine and BlockwiseV2 scales and zero points are in B's
  // layout. For regular dots, they need to be transposed like B (`adj_y`). For
  // blockwise dots, ynn::define_blockwise_dot expects [..., N, num_blocks],
  // which is the opposite of B's layout after `adj_y`.
  if (weight_quant.has_weight_layout && (is_blockwise ? !adj_y : adj_y)) {
    TF_LITE_ENSURE_STATUS(transpose(b_scale_id));
    TF_LITE_ENSURE_STATUS(transpose(b_zp_id));
  }

  if (mutual_broadcast) {
    TF_LITE_ENSURE_STATUS(ImplementMutualBroadcasting(
        context, subgraph, rank_a, rank_b, /*exclude_a=*/2, /*exclude_b=*/2,
        current_a_id, current_b_id));
    rank_a = std::max(rank_a, rank_b);
    rank_b = rank_a;
  }

  uint32_t dot_flags = 0;
  if (input_b_tensor.type == kTfLiteInt8 && IsConstant(input_b_tensor) &&
      b_zp_id == YNN_INVALID_VALUE_ID) {
    // In TFlite, quantized int8 weights are in the range [-127, 127] (see
    // quantization_spec.md for reference).
    dot_flags |= YNN_NODE_FLAG_SYMMETRIC_B;
  }

  if (is_blockwise) {
    if (bias_id != YNN_INVALID_VALUE_ID && bias_tensor_index >= 0) {
      uint32_t float_bias_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(
          DequantizeIfNeeded(context, subgraph, tensor_to_value_id,
                             bias_tensor_index, bias_id, &float_bias_id));
      bias_id = float_bias_id;
    }

    ynn_type output_ynn_type = GetYnnType(output_tensor.type);
    uint32_t dot_output_id =
        is_output_quantized ? YNN_INVALID_VALUE_ID : *output_id;
    TF_LITE_ENSURE_YNN_STATUS(ynn::define_blockwise_dot(
        subgraph, current_a_id, a_zp_id, a_scale_id, current_b_id, b_zp_id,
        b_scale_id, weight_quant.block_size, bias_id,
        is_output_quantized ? ynn_type_fp32 : output_ynn_type, dot_output_id,
        dot_flags));

    if (is_output_quantized) {
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_quantize(subgraph, dot_output_id, output_ynn_type,
                              out_zp_id, out_scale_id, output_id, 0));
    } else {
      *output_id = dot_output_id;
    }
    return kTfLiteOk;
  }

  if (is_quantized || is_dynamically_quantized) {
    // Blockwise weights do not share a single scale with the dot accumulator,
    // so quantized biases are dequantized and added in float.
    const bool bias_uses_float_add =
        is_dynamically_quantized ||
        input_b_tensor.quantization.type == kTfLiteBlockwiseQuantization ||
        input_b_tensor.quantization.type == kTfLiteBlockwiseQuantizationV2;
    uint32_t int_bias_id = bias_uses_float_add ? YNN_INVALID_VALUE_ID : bias_id;
    uint32_t float_bias_id = YNN_INVALID_VALUE_ID;
    if (bias_uses_float_add && bias_id != YNN_INVALID_VALUE_ID) {
      if (bias_tensor_index >= 0) {
        TF_LITE_ENSURE_STATUS(
            DequantizeIfNeeded(context, subgraph, tensor_to_value_id,
                               bias_tensor_index, bias_id, &float_bias_id));
      } else {
        float_bias_id = bias_id;
      }
    }

    TF_LITE_ENSURE_STATUS(DefineQuantizedDot(
        context, subgraph, /*num_k_dims=*/1, current_a_id, a_scale_id, a_zp_id,
        current_b_id, b_scale_id, b_zp_id, int_bias_id, out_scale_id, out_zp_id,
        GetYnnType(output_tensor.type), output_id, dot_flags, float_bias_id));
  } else {
    if (*output_id == YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_tensor(subgraph, GetYnnType(output_tensor.type), 0,
                            nullptr, nullptr, 0, output_id));
    }
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1,
                                             current_a_id, current_b_id,
                                             bias_id, output_id, 0));
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteStatus IsBatchMatMulSupported(const TfLiteRegistration* registration,
                                    const TfLiteNode* node,
                                    TfLiteContext* context,
                                    const TfLiteYNNPackDelegateOptions& options,
                                    bool is_runtime_bmm) {
  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input_a = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& input_b = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, tflite::NumElements(&input_a) > 0);
  TF_LITE_ENSURE(context, tflite::NumElements(&input_b) > 0);

  TF_LITE_ENSURE(context, IsTensorSupported(input_a));
  TF_LITE_ENSURE(context, IsTensorTypeSupported(input_b));
  TF_LITE_ENSURE(context, IsTensorSupported(output));

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  if (input_a.type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, input_b.type == kTfLiteInt8 ||
                                input_b.type == kTfLiteInt4 ||
                                input_b.type == kTfLiteInt2);
    TF_LITE_ENSURE(context,
                   output.type == kTfLiteInt8 || output.type == kTfLiteInt32);
  } else if (is_float_type(input_a.type)) {
    if (!is_float_type(input_b.type)) {
      TF_LITE_ENSURE(context, input_b.type == kTfLiteInt8 ||
                                  input_b.type == kTfLiteInt4 ||
                                  input_b.type == kTfLiteInt2);
    }
    TF_LITE_ENSURE(context, is_float_type(output.type));
  } else {
    return kTfLiteError;
  }

  if (input_b.type == kTfLiteInt4) {
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

  bool adj_y = false;
  if (!is_runtime_bmm) {
    const auto* params =
        static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);
    if (params) adj_y = params->adj_y;
  } else {
    adj_y = true;
  }
  const int k_axis = adj_y ? input_b.dims->size - 1 : input_b.dims->size - 2;
  MatMulWeightQuantization weight_quant;
  TF_LITE_ENSURE_STATUS(GetMatMulWeightQuantization(context, input_b, k_axis,
                                                    &options, &weight_quant));

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

TfLiteStatus IsRuntimeBatchedMatMulSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context, const TfLiteYNNPackDelegateOptions& options) {
  TF_LITE_ENSURE(context, IsRuntimeBmm(registration, node));

  return IsBatchMatMulSupported(registration, node, context, options,
                                /*is_runtime_bmm=*/true);
}

TfLiteStatus IsFullyConnectedSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context, const TfLiteYNNPackDelegateOptions& options) {
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& weights = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, tflite::NumElements(&input) > 0);
  TF_LITE_ENSURE(context, tflite::NumElements(&weights) > 0);

  bool has_bias = node->inputs->size == 3 && node->inputs->data[2] >= 0;

  TF_LITE_ENSURE(context, IsTensorSupported(input));
  TF_LITE_ENSURE(context, IsTensorTypeSupported(weights));
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
                                weights.type == kTfLiteInt2);
    TF_LITE_ENSURE(context, is_float_type(output.type));
    if (has_bias) {
      const TfLiteTensor& bias = context->tensors[node->inputs->data[2]];
      TF_LITE_ENSURE(context, is_float_type(bias.type));
    }
  } else {
    return kTfLiteError;
  }

  if (weights.type == kTfLiteInt4) {
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

  MatMulWeightQuantization weight_quant;
  TF_LITE_ENSURE_STATUS(GetMatMulWeightQuantization(
      context, weights, /*k_axis=*/1, &options, &weight_quant));

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

  const auto* params =
      static_cast<const TfLiteBatchMatMulParams*>(node.builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int rank_a = input_a_tensor.dims->size;
  int rank_b = input_b_tensor.dims->size;

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  TF_LITE_ENSURE_STATUS(DefineMatMul(
      context, subgraph, tensor_to_value_id, rank_a, rank_b, input_a_val_id,
      input_b_val_id, YNN_INVALID_VALUE_ID, params->adj_x, params->adj_y,
      /*mutual_broadcast=*/true, params->asymmetric_quantize_inputs,
      input_a_tensor, input_b_tensor, output_tensor, &output_val_id));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}


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
  bool asymmetric_quantize_inputs = true;
  bool is_src = false;
  if (node.builtin_code == kTfLiteBuiltinBatchMatmul) {
    const auto* params =
        static_cast<const TfLiteBatchMatMulParams*>(node.builtin_data);
    if (params) {
      adj_x = params->adj_x;
      adj_y = params->adj_y;
      asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;
    }
  } else {
    const flexbuffers::Map flexbuffer_map = GetFlexBufferMap(node);
    if (!flexbuffer_map["is_src"].IsNull()) {
      is_src = flexbuffer_map["is_src"].AsBool();
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
          context, subgraph, tensor_to_value_id, rank_a, rank_b,
          current_a_val_id, current_b_val_id, YNN_INVALID_VALUE_ID, adj_x,
          /*adj_y=*/true, /*mutual_broadcast=*/true, asymmetric_quantize_inputs,
          input_a_tensor, input_b_tensor, output_tensor, &full_matmul_val_id));

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
          context, subgraph, tensor_to_value_id, rank_a, rank_b,
          current_a_val_id, current_b_val_id, YNN_INVALID_VALUE_ID,
          /*adj_x=*/false, /*adj_y=*/true, /*mutual_broadcast=*/true,
          asymmetric_quantize_inputs, input_a_tensor, input_b_tensor,
          output_tensor, &output_val_id));
    }
  } else {
    TF_LITE_ENSURE_STATUS(DefineMatMul(
        context, subgraph, tensor_to_value_id, rank_a, rank_b, current_a_val_id,
        current_b_val_id, YNN_INVALID_VALUE_ID, adj_x, adj_y,
        /*mutual_broadcast=*/true, asymmetric_quantize_inputs, input_a_tensor,
        input_b_tensor, output_tensor, &output_val_id));
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

  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node.builtin_data);
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

  TF_LITE_ENSURE_STATUS(DefineMatMul(
      context, subgraph, tensor_to_value_id, rank_a, rank_b,
      reshaped_input_val_id, weights_val_id, bias_val_id,
      /*adj_x=*/false, /*adj_y=*/true,
      /*mutual_broadcast=*/false, params->asymmetric_quantize_inputs,
      input_tensor, weights_tensor, output_tensor, &matmul_output_id,
      bias_tensor_index));

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
  if (is_quantized &&
      filter_tensor.quantization.type == kTfLiteAffineQuantization) {
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

  if (is_per_channel) {
    // Reshape filter scales and zero points to match dot output channels:
    // [co], or [g, 1, co/g] for grouped convolutions.
    const size_t dot_dims[] = {groups, 1, group_output_channels};
    const size_t rank = (groups == 1) ? 1 : 3;
    const size_t* dims = (groups == 1) ? &dot_dims[2] : dot_dims;
    for (uint32_t* id : {&current_b_scale_id, &current_b_zp_id}) {
      if (*id == YNN_INVALID_VALUE_ID) continue;
      uint32_t reshaped_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
          subgraph, rank, dims, *id, &reshaped_id, 0));
      *id = reshaped_id;
    }
  }

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
    uint32_t dot_flags = 0;
    if (filter_tensor.type == kTfLiteInt8 && IsConstant(filter_tensor) &&
        current_b_zp_id == YNN_INVALID_VALUE_ID) {
      // In TFlite, quantized int8 weights are in the range [-127, 127] (see
      // quantization_spec.md for reference).
      dot_flags |= YNN_NODE_FLAG_SYMMETRIC_B;
    }

    TF_LITE_ENSURE_STATUS(DefineQuantizedDot(
        context, subgraph, /*num_k_dims=*/3, current_input_id, a_scale_id,
        a_zp_id, transposed_filter_id, current_b_scale_id, current_b_zp_id,
        current_bias_id, out_scale_id, out_zp_id,
        GetYnnType(output_tensor.type), &dot_output_id, dot_flags));
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

  const auto* params = static_cast<const TfLiteConvParams*>(node.builtin_data);
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

  const auto* params =
      static_cast<const TfLiteDepthwiseConvParams*>(node.builtin_data);
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
