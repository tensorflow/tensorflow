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

#include "tensorflow/lite/delegates/ynnpack/attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
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

struct SdpaInputs {
  int q_index = -1;
  int k_index = -1;
  int v_index = -1;
  int mask_index = -1;
  int param_index = -1;
};

SdpaInputs GetSdpaInputs(TfLiteContext* context, const NodeInfo& node) {
  SdpaInputs inputs;
  inputs.q_index = node.inputs[0];
  inputs.k_index = node.inputs[1];
  inputs.v_index = node.inputs[2];

  if (node.inputs.size() >= 4) {
    int idx3 = node.inputs[3];
    if (idx3 != -1) {
      const TfLiteTensor& tensor3 = context->tensors[idx3];
      if (tensor3.type == kTfLiteInt32 || tensor3.type == kTfLiteInt64) {
        inputs.param_index = idx3;
      } else {
        inputs.mask_index = idx3;
      }
    }
  }
  if (node.inputs.size() >= 5) {
    int idx4 = node.inputs[4];
    if (idx4 != -1) {
      inputs.param_index = idx4;
    }
  }
  return inputs;
}

bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
         type == kTfLiteBFloat16;
}

bool IsSdpaDotTypeSupported(TfLiteType type) {
  return type == kTfLiteNoType || type == kTfLiteInt8 || IsFloatType(type);
}

// Returns the more precise of the types `a` and `b` of the operands of a dot.
// Any type that is not a floating point type is a quantized type.
TfLiteType MorePreciseType(TfLiteType a, TfLiteType b) {
  return TfLiteTypeGetSizeBits(a) >= TfLiteTypeGetSizeBits(b) ? a : b;
}

// Computes the types of the operands of the dots Q @ K^T and P @ V for the
// delegate option `sdpa_dot_type`. `kTfLiteInt8` indicates the operands are
// multiplied as quantized int8.
void GetSdpaDotTypes(TfLiteType sdpa_dot_type, TfLiteType q_type,
                     bool q_is_quantized, TfLiteType k_type,
                     bool k_is_quantized, TfLiteType v_type,
                     bool v_is_quantized, TfLiteType& qk_dot_type,
                     TfLiteType& pv_dot_type) {
  if (q_is_quantized) q_type = kTfLiteInt8;
  if (k_is_quantized) k_type = kTfLiteInt8;
  if (v_is_quantized) v_type = kTfLiteInt8;
  if (sdpa_dot_type == kTfLiteNoType) {
    qk_dot_type = MorePreciseType(q_type, k_type);
    // P is computed in float32, unless Q @ K^T is computed in a less precise
    // floating point type.
    const TfLiteType p_type =
        IsFloatType(qk_dot_type) ? qk_dot_type : kTfLiteFloat32;
    pv_dot_type = MorePreciseType(p_type, v_type);
  } else if (sdpa_dot_type == kTfLiteInt8) {
    // Q and P are quantized or converted to the type of K and V.
    qk_dot_type = k_type;
    pv_dot_type = v_type;
  } else {
    qk_dot_type = sdpa_dot_type;
    pv_dot_type = sdpa_dot_type;
  }
}

// Converts `*id`, the value of `tensor`, to the floating point type `type`,
// dequantizing it if `tensor` is quantized.
TfLiteStatus DefineSdpaFloatOperand(TfLiteContext* context,
                                    ynn_subgraph_t subgraph,
                                    const TfLiteTensor& tensor, ynn_type type,
                                    uint32_t* id) {
  uint32_t float_id = YNN_INVALID_VALUE_ID;
  if (IsQuantized(tensor)) {
    uint32_t scale_id = YNN_INVALID_VALUE_ID;
    uint32_t zp_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(
        DefineQuantizationParams(context, subgraph, tensor, &scale_id, &zp_id));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dequantize(
        subgraph, *id, zp_id, scale_id, type, &float_id, 0));
  } else if (GetYnnType(tensor.type) != type) {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_convert(subgraph, *id, type, &float_id, 0));
  } else {
    return kTfLiteOk;
  }
  *id = float_id;
  return kTfLiteOk;
}

// Computes `output = a @ b` (or `(b @ a^T)^T` when `transpose_ab` is true)
// where `a` and `b` are float or per-tensor quantized int8.
// When `transpose_ab` is false, `a` is [..., M, K] and `b` is [..., K, N].
// When `transpose_ab` is true, `a` is [..., M, K] and `b` is [..., N, K]
// (already contiguous along K), avoiding transposing/packing `b` when M is
// small.
// If `dot_type` is a floating point type, `a` must be a value of that type,
// and `b` is dequantized or converted to it. Otherwise, `b` must be quantized,
// and if `a` is float, it is dynamically quantized to int8 along its reduction
// axis (-1).
// When `output_quant_tensor` is non-null, quantizes the result to it.
TfLiteStatus DefineSdpaDot(TfLiteContext* context, ynn_subgraph_t subgraph,
                           ynn_type dot_type, uint32_t a_id,
                           bool a_is_quantized, uint32_t a_scale_id,
                           uint32_t a_zp_id, uint32_t b_id,
                           const TfLiteTensor& b_tensor,
                           const TfLiteTensor* output_quant_tensor,
                           ynn_type output_type, uint32_t* output_id,
                           bool transpose_ab = false) {
  const int32_t swap_last_two_perm[] = {-1, -2};
  auto transpose_last_two = [&](uint32_t& id) -> TfLiteStatus {
    if (id == YNN_INVALID_VALUE_ID) return kTfLiteOk;
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 2, swap_last_two_perm, id,
                                    &transposed_id, YNN_NODE_FLAG_KEEP_DIMS));
    id = transposed_id;
    return kTfLiteOk;
  };

  if (dot_type != ynn_type_int8) {
    TF_LITE_ENSURE_STATUS(
        DefineSdpaFloatOperand(context, subgraph, b_tensor, dot_type, &b_id));
    if (transpose_ab) {
      TF_LITE_ENSURE_STATUS(transpose_last_two(a_id));
      std::swap(a_id, b_id);
    }
    if (output_quant_tensor == nullptr) {
      uint32_t local_out_id = YNN_INVALID_VALUE_ID;
      uint32_t* dot_out_id = transpose_ab ? &local_out_id : output_id;
      if (*dot_out_id == YNN_INVALID_VALUE_ID) {
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
            subgraph, output_type, 0, nullptr, nullptr, 0, dot_out_id));
      }
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_id,
                                               b_id, YNN_INVALID_VALUE_ID,
                                               dot_out_id, 0));
      if (transpose_ab) {
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
            subgraph, 2, swap_last_two_perm, local_out_id, output_id,
            YNN_NODE_FLAG_KEEP_DIMS));
      }
      return kTfLiteOk;
    }

    uint32_t float_out_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
        subgraph, ynn_type_fp32, 0, nullptr, nullptr, 0, &float_out_id));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_id,
                                             b_id, YNN_INVALID_VALUE_ID,
                                             &float_out_id, 0));
    if (transpose_ab) {
      TF_LITE_ENSURE_STATUS(transpose_last_two(float_out_id));
    }

    uint32_t out_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t out_zp_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
        context, subgraph, *output_quant_tensor, &out_scale_id, &out_zp_id));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(subgraph, float_out_id,
                                                  output_type, out_zp_id,
                                                  out_scale_id, output_id, 0));
    return kTfLiteOk;
  }

  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(context, subgraph, b_tensor,
                                                 &b_scale_id, &b_zp_id));

  uint32_t q_a_id = a_id;
  uint32_t q_a_zp_id = a_zp_id;
  uint32_t q_a_scale_id = a_scale_id;
  if (!a_is_quantized) {
    int32_t reduce_axis = -1;
    uint32_t min_max_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
        subgraph, ynn_reduce_min_max, 1, &reduce_axis, a_id,
        YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS));

    q_a_zp_id = YNN_INVALID_VALUE_ID;
    q_a_scale_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dynamic_quantization(
        subgraph, min_max_id, ynn_type_int8, &q_a_zp_id, &q_a_scale_id, 0));

    q_a_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(
        subgraph, a_id, ynn_type_int8, q_a_zp_id, q_a_scale_id, &q_a_id, 0));
  }

  if (transpose_ab) {
    TF_LITE_ENSURE_STATUS(transpose_last_two(q_a_id));
    if (!a_is_quantized) {
      // Dynamically quantized `a` has per-row [..., M, 1] zero points that
      // need to become [..., 1, M] to match `a^T` on the RHS. `q_a_scale_id` is
      // kept un-transposed because we transpose `accum_id` back to [..., M, N]
      // before dequantizing.
      TF_LITE_ENSURE_STATUS(transpose_last_two(q_a_zp_id));
    }
    std::swap(q_a_id, b_id);
    std::swap(q_a_zp_id, b_zp_id);
  }

  uint32_t dot_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_dot_quantization(
      subgraph, /*num_k_dims=*/1, q_a_id, q_a_zp_id, q_a_scale_id, b_id,
      b_zp_id, b_scale_id, dot_zp_id, dot_scale_id));

  uint32_t accum_init_id = YNN_INVALID_VALUE_ID;
  if (dot_zp_id != YNN_INVALID_VALUE_ID) {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(subgraph, ynn_unary_negate,
                                               dot_zp_id, &accum_init_id, 0));
  }

  uint32_t accum_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, q_a_id,
                                           b_id, accum_init_id, &accum_id, 0));

  if (transpose_ab) {
    // Transpose the int32 accumulator from [..., N, M] to [..., M, N] before
    // dequantizing so the subsequent scale/quantize ops vectorize over the
    // larger N dimension instead of the small M dimension.
    TF_LITE_ENSURE_STATUS(transpose_last_two(accum_id));
  }

  if (output_quant_tensor == nullptr) {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                              dot_scale_id, output_type, output_id, 0));
    return kTfLiteOk;
  }

  uint32_t float_out_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                            dot_scale_id, ynn_type_fp32, &float_out_id, 0));
  uint32_t out_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t out_zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, *output_quant_tensor, &out_scale_id, &out_zp_id));
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(subgraph, float_out_id,
                                                output_type, out_zp_id,
                                                out_scale_id, output_id, 0));
  return kTfLiteOk;
}

}  // namespace

bool IsSdpa(const TfLiteRegistration* registration, const TfLiteNode* node) {
  if (registration == nullptr) {
    return false;
  }
  if (registration->builtin_code == kTfLiteBuiltinCustom &&
      registration->custom_name != nullptr) {
    if (strcmp(registration->custom_name,
               "odml.scaled_dot_product_attention") == 0 ||
        strcmp(registration->custom_name, "odml.sdpa_transposed") == 0) {
      return true;
    }
  }
  if (registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node != nullptr && node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    if (composite_params->name != nullptr) {
      if (strcmp(composite_params->name, "odml.scaled_dot_product_attention") ==
              0 ||
          strcmp(composite_params->name, "odml.sdpa_transposed") == 0) {
        return true;
      }
    }
  }
  return false;
}

TfLiteStatus IsSdpaSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context,
                             const TfLiteYNNPackDelegateOptions& options) {
  TF_LITE_ENSURE(context, IsSdpa(registration, node));
  TF_LITE_ENSURE(context, node->inputs->size >= 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& q = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& k = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& v = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };
  auto is_quantized_int8 = [](const TfLiteTensor& tensor) {
    return tensor.type == kTfLiteInt8 && IsQuantized(tensor);
  };
  auto is_quantized_int4_or_int8 = [](const TfLiteTensor& tensor) {
    return (tensor.type == kTfLiteInt8 || tensor.type == kTfLiteInt4) &&
           IsQuantized(tensor);
  };

  TF_LITE_ENSURE(context, IsSdpaDotTypeSupported(options.sdpa_dot_type));

  TF_LITE_ENSURE(context, IsTensorSupported(q));
  TF_LITE_ENSURE(context,
                 is_float_type(q.type) || is_quantized_int4_or_int8(q));
  TF_LITE_ENSURE(context, IsTensorSupported(k));
  TF_LITE_ENSURE(context,
                 is_float_type(k.type) || is_quantized_int4_or_int8(k));
  if (!IsQuantized(q) && !IsQuantized(k)) {
    TF_LITE_ENSURE_EQ(context, k.type, q.type);
  }
  TF_LITE_ENSURE(context, IsTensorSupported(v));
  TF_LITE_ENSURE(context,
                 is_float_type(v.type) || is_quantized_int4_or_int8(v));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  TF_LITE_ENSURE(context,
                 is_float_type(output.type) || is_quantized_int8(output));

  TF_LITE_ENSURE_EQ(context, q.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, k.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, v.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output.dims->size, 4);

  bool is_seq_major = true;
  if (registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    if (composite_params->name != nullptr &&
        strcmp(composite_params->name, "odml.sdpa_transposed") == 0) {
      is_seq_major = false;
    }
  } else if (registration->builtin_code == kTfLiteBuiltinCustom &&
             registration->custom_name != nullptr &&
             strcmp(registration->custom_name, "odml.sdpa_transposed") == 0) {
    is_seq_major = false;
  }
  if (is_seq_major) {
    TF_LITE_ENSURE_EQ(context, q.dims->data[2], k.dims->data[2]);
    TF_LITE_ENSURE_EQ(context, k.dims->data[2], v.dims->data[2]);
  } else {
    TF_LITE_ENSURE(context, k.dims->data[1] > 0);
    TF_LITE_ENSURE_EQ(context, k.dims->data[1], v.dims->data[1]);
    TF_LITE_ENSURE(context, q.dims->data[1] >= k.dims->data[1] &&
                                q.dims->data[1] % k.dims->data[1] == 0);
  }

  // If 4th input is present, it can be Mask or Param.
  if (node->inputs->size >= 4 && node->inputs->data[3] != -1) {
    const TfLiteTensor& input3 = context->tensors[node->inputs->data[3]];
    TF_LITE_ENSURE(context,
                   input3.type == kTfLiteBool || IsTensorSupported(input3));
    if (is_float_type(input3.type) || input3.type == kTfLiteBool ||
        is_quantized_int4_or_int8(input3)) {
      // If it is Mask, it should be float, bool, or quantized int4/int8.
    } else if (input3.type == kTfLiteInt32 || input3.type == kTfLiteInt64) {
      TF_LITE_ENSURE(context, !IsQuantized(input3));
    } else {
      return kTfLiteError;
    }
  }

  // If 5th input is present, it must be Param.
  if (node->inputs->size >= 5 && node->inputs->data[4] != -1) {
    const TfLiteTensor& input4 = context->tensors[node->inputs->data[4]];
    TF_LITE_ENSURE(context, IsTensorSupported(input4));
    TF_LITE_ENSURE(context,
                   input4.type == kTfLiteInt32 || input4.type == kTfLiteInt64);
    TF_LITE_ENSURE(context, !IsQuantized(input4));
  }

  return kTfLiteOk;
}

TfLiteStatus DefineSdpaNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            uint32_t& next_external_id,
                            std::vector<DummyInputInfo>& dummy_inputs,
                            const NodeInfo& node,
                            const TfLiteYNNPackDelegateOptions& options) {
  SdpaInputs sdpa_inputs = GetSdpaInputs(context, node);

  const TfLiteTensor& q_tensor = context->tensors[sdpa_inputs.q_index];
  const TfLiteTensor& k_tensor = context->tensors[sdpa_inputs.k_index];
  const TfLiteTensor& v_tensor = context->tensors[sdpa_inputs.v_index];
  const TfLiteTensor& output_tensor = context->tensors[node.outputs[0]];

  // The types of the operands of the dots Q @ K^T and P @ V.
  TfLiteType qk_dot_type = kTfLiteNoType;
  TfLiteType pv_dot_type = kTfLiteNoType;
  GetSdpaDotTypes(options.sdpa_dot_type, q_tensor.type, IsQuantized(q_tensor),
                  k_tensor.type, IsQuantized(k_tensor), v_tensor.type,
                  IsQuantized(v_tensor), qk_dot_type, pv_dot_type);
  const bool is_float_qk_dot = IsFloatType(qk_dot_type);
  const bool is_float_pv_dot = IsFloatType(pv_dot_type);

  uint32_t q_val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         sdpa_inputs.q_index);
  uint32_t k_val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         sdpa_inputs.k_index);
  uint32_t v_val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         sdpa_inputs.v_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, node.outputs[0]);
  uint32_t mask_val_id = YNN_INVALID_VALUE_ID;
  if (sdpa_inputs.mask_index != -1) {
    mask_val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                     sdpa_inputs.mask_index);
  }

  TF_LITE_ENSURE(context, q_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, k_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, v_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  bool is_seq_major = true;
  if (node.builtin_code == kTfLiteBuiltinStablehloComposite) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node.builtin_data);
    if (composite_params != nullptr && composite_params->name != nullptr) {
      if (strcmp(composite_params->name, "odml.sdpa_transposed") == 0) {
        is_seq_major = false;
      }
    }
  } else if (node.builtin_code == kTfLiteBuiltinCustom &&
             node.custom_name != nullptr) {
    if (strcmp(node.custom_name, "odml.sdpa_transposed") == 0) {
      is_seq_major = false;
    }
  }

  float scale_val = 1.0f;
  bool scale_specified = false;
  const flexbuffers::Map flexbuffer_map = GetFlexBufferMap(node);
  if (!flexbuffer_map["scale"].IsNull()) {
    scale_val = flexbuffer_map["scale"].AsFloat();
    scale_specified = true;
  }

  if (!scale_specified) {
    if (is_seq_major) {
      scale_val = 1.0f / std::sqrt(static_cast<float>(q_tensor.dims->data[3]));
    } else {
      scale_val = 1.0f;
    }
  }

  float logit_cap_val = 0.0f;
  bool has_logit_cap = false;
  if (!flexbuffer_map["logit_cap"].IsNull()) {
    logit_cap_val = flexbuffer_map["logit_cap"].AsFloat();
    has_logit_cap = logit_cap_val > 0.0f;
  }

  const int k_seq_axis = is_seq_major ? 1 : 2;
  const int v_seq_axis = is_seq_major ? 1 : 3;
  const int q_seq_axis = is_seq_major ? 1 : 2;

  uint32_t current_q_val_id = q_val_id;
  if (q_tensor.type == kTfLiteInt4) {
    uint32_t q_int8_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(subgraph, current_q_val_id,
                                                 ynn_type_int8, &q_int8_id, 0));
    current_q_val_id = q_int8_id;
  }
  uint32_t current_k_val_id = k_val_id;
  uint32_t current_v_val_id = v_val_id;
  uint32_t current_mask_val_id = mask_val_id;

  if (sdpa_inputs.param_index != -1) {
    // Slice K to the current kv-seq length. The length is carried by a dummy
    // external input whose seq axis is resized from the param tensor at each
    // invoke.
    size_t full_dims_k[YNN_MAX_TENSOR_RANK];
    for (int i = 0; i < k_tensor.dims->size; ++i) {
      full_dims_k[i] = k_tensor.dims->data[i];
    }
    uint32_t dummy_val_id_k = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
        context, subgraph, next_external_id, dummy_inputs,
        sdpa_inputs.param_index, k_seq_axis, k_tensor.dims->size, full_dims_k,
        ynn_type_fp32, &dummy_val_id_k));

    uint32_t sliced_k_val_id = YNN_INVALID_VALUE_ID;
    int32_t slice_axes_k[1] = {k_seq_axis};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
        subgraph, /*num_axes=*/1, slice_axes_k, current_k_val_id,
        dummy_val_id_k, &sliced_k_val_id, /*flags=*/0));
    current_k_val_id = sliced_k_val_id;
  }
  if (k_tensor.type == kTfLiteInt4) {
    uint32_t k_int8_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(subgraph, current_k_val_id,
                                                 ynn_type_int8, &k_int8_id, 0));
    current_k_val_id = k_int8_id;
  }

  // Bring Q and K to [B, H, S, D]. V is handled below, next to the dot that
  // consumes it.
  uint32_t q_trans_id = YNN_INVALID_VALUE_ID;
  uint32_t k_trans_id = YNN_INVALID_VALUE_ID;

  if (is_seq_major) {
    const int32_t io_perm[] = {0, 2, 1, 3};

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, current_q_val_id, &q_trans_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, current_k_val_id, &k_trans_id, 0));
  } else {
    // For sdpa_transposed, Q and K are already in correct layout.
    q_trans_id = current_q_val_id;
    k_trans_id = current_k_val_id;
  }

  const int q_head_dim = is_seq_major ? 2 : 1;
  const int k_head_dim = is_seq_major ? 2 : 1;
  const int n_q = q_tensor.dims->data[q_head_dim];
  const int n_kv = k_tensor.dims->data[k_head_dim];
  TF_LITE_ENSURE(context, n_kv > 0 && n_q % n_kv == 0);
  const size_t g_heads_per_kv = static_cast<size_t>(n_q / n_kv);

  const int q_seq_dim = is_seq_major ? 1 : 2;
  const int q_seq = q_tensor.dims->data[q_seq_dim];
  const bool is_small_q =
      !is_seq_major && (q_seq * static_cast<int>(g_heads_per_kv) <= 8);
  // Prefill with grouped heads: keep the head axis as [n_kv, g] and let the dot
  // broadcast K/V over g, instead of folding g into the row axis.
  const bool gqa_batch = g_heads_per_kv > 1 && !is_seq_major && !is_small_q;
  const bool gqa_fold = g_heads_per_kv > 1 && !gqa_batch;

  if (g_heads_per_kv > 1) {
    const size_t q_splits[2] = {static_cast<size_t>(n_kv), g_heads_per_kv};
    uint32_t q_5d_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(subgraph, /*axis=*/1,
                                                   /*num_splits=*/2, q_splits,
                                                   q_trans_id, &q_5d_id, 0));
    if (gqa_fold) {
      uint32_t q_packed_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
          subgraph, /*axis=*/2, /*axes_count=*/2, q_5d_id, &q_packed_id, 0));
      q_trans_id = q_packed_id;
    } else {
      q_trans_id = q_5d_id;
      const int32_t expand_axis = 2;
      uint32_t k_5d_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
          subgraph, /*num_new_axes=*/1, &expand_axis, k_trans_id, &k_5d_id, 0));
      k_trans_id = k_5d_id;
    }
  }

  bool need_slice_out = false;
  uint32_t post_bmm_id = YNN_INVALID_VALUE_ID;
  uint32_t* post_bmm_ptr = &post_bmm_id;

  if (!need_slice_out && !is_seq_major && g_heads_per_kv == 1) {
    post_bmm_ptr = &output_val_id;
  }

  // Scale Q.
  const bool q_is_quantized = IsQuantized(q_tensor);
  uint32_t q_scaled_id = q_trans_id;
  uint32_t q_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t q_zp_id = YNN_INVALID_VALUE_ID;
  bool bmm1_a_is_quantized = false;
  if (q_is_quantized) {
    TF_LITE_ENSURE_STATUS(DefineQuantizationParams(context, subgraph, q_tensor,
                                                   &q_scale_id, &q_zp_id));
    if (scale_val != 1.0f) {
      uint32_t scale_const_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &scale_val,
                            YNN_VALUE_FLAG_COPY_DATA_FP32, &scale_const_id));
      uint32_t scaled_q_scale_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                  q_scale_id, scale_const_id,
                                                  &scaled_q_scale_id, 0));
      q_scale_id = scaled_q_scale_id;
    }
    if (!is_float_qk_dot) {
      bmm1_a_is_quantized = true;
    } else {
      q_scaled_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_dequantize(subgraph, q_trans_id, q_zp_id, q_scale_id,
                                GetYnnType(qk_dot_type), &q_scaled_id, 0));
    }
  } else if (scale_val != 1.0f) {
    uint32_t scale_const_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
        subgraph, GetYnnType(q_tensor.type), 0, nullptr, &scale_val,
        YNN_VALUE_FLAG_COPY_DATA_FP32, &scale_const_id));
    q_scaled_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                q_trans_id, scale_const_id,
                                                &q_scaled_id, 0));
  }
  if (is_float_qk_dot && !q_is_quantized && q_tensor.type != qk_dot_type) {
    uint32_t q_converted_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(
        subgraph, q_scaled_id, GetYnnType(qk_dot_type), &q_converted_id, 0));
    q_scaled_id = q_converted_id;
  }

  // Scores: S = Q @ K^T, [B, H, Q, S].
  // When `is_small_q`, compute S = (K @ Q^T)^T to avoid transposing/packing K.
  const int32_t swap_last_two_perm[] = {-1, -2};
  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  uint32_t bmm1_b_id = k_trans_id;
  if (!is_small_q) {
    uint32_t k_trans_t_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 2, swap_last_two_perm, k_trans_id,
                                    &k_trans_t_id, YNN_NODE_FLAG_KEEP_DIMS));
    bmm1_b_id = k_trans_t_id;
  }
  TF_LITE_ENSURE_STATUS(DefineSdpaDot(
      context, subgraph, GetYnnType(qk_dot_type), q_scaled_id,
      bmm1_a_is_quantized, q_scale_id, q_zp_id, bmm1_b_id, k_tensor,
      /*output_quant_tensor=*/nullptr, ynn_type_fp32, &scores_id,
      /*transpose_ab=*/is_small_q));

  uint32_t logits_id = scores_id;
  if (has_logit_cap) {
    uint32_t cap_const_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &logit_cap_val,
                          YNN_VALUE_FLAG_COPY_DATA_FP32, &cap_const_id));

    uint32_t scores_div_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_divide,
                                                scores_id, cap_const_id,
                                                &scores_div_id, 0));

    uint32_t scores_tanh_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(
        subgraph, ynn_unary_tanh, scores_div_id, &scores_tanh_id, 0));

    uint32_t scores_capped_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                scores_tanh_id, cap_const_id,
                                                &scores_capped_id, 0));
    logits_id = scores_capped_id;
  }

  uint32_t masked_logits_id = YNN_INVALID_VALUE_ID;
  if (current_mask_val_id != YNN_INVALID_VALUE_ID) {
    const TfLiteTensor& mask_tensor = context->tensors[sdpa_inputs.mask_index];
    uint32_t mask_to_add_id = current_mask_val_id;
    if (sdpa_inputs.param_index != -1) {
      // Slice the mask along its kv-seq (last) axis using the logits as the
      // template.
      int32_t mask_seq_axis = mask_tensor.dims->size - 1;
      uint32_t sliced_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, /*num_axes=*/1, &mask_seq_axis, mask_to_add_id, logits_id,
          &sliced_mask_id, /*flags=*/0));
      mask_to_add_id = sliced_mask_id;
    }
    if (mask_tensor.type == kTfLiteBool) {
      uint32_t mask_float_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(
          subgraph, mask_to_add_id, ynn_type_fp32, &mask_float_id, 0));

      float one_val = 1.0f;
      uint32_t one_const_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &one_val,
                            YNN_VALUE_FLAG_COPY_DATA_FP32, &one_const_id));

      float mask_fill_val = -10000.0f;
      uint32_t mask_fill_const_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
          subgraph, ynn_type_fp32, 0, nullptr, &mask_fill_val,
          YNN_VALUE_FLAG_COPY_DATA_FP32, &mask_fill_const_id));

      uint32_t inv_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_subtract,
                                                  one_const_id, mask_float_id,
                                                  &inv_mask_id, 0));

      uint32_t float_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_binary(subgraph, ynn_binary_multiply, inv_mask_id,
                            mask_fill_const_id, &float_mask_id, 0));

      mask_to_add_id = float_mask_id;
    } else if (IsQuantized(mask_tensor)) {
      if (mask_tensor.type == kTfLiteInt4) {
        uint32_t mask_int8_id = YNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(
            subgraph, mask_to_add_id, ynn_type_int8, &mask_int8_id, 0));
        mask_to_add_id = mask_int8_id;
      }
      uint32_t mask_scale_id = YNN_INVALID_VALUE_ID;
      uint32_t mask_zp_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
          context, subgraph, mask_tensor, &mask_scale_id, &mask_zp_id));
      uint32_t float_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_dequantize(
          subgraph, mask_to_add_id, mask_zp_id, mask_scale_id, ynn_type_fp32,
          &float_mask_id, 0));
      mask_to_add_id = float_mask_id;
    }
    // When gqa_fold is used with T == 1 (or any query-independent [B, 1, 1, S]
    // mask), the 4D mask already broadcasts across the folded [g * T] axis, so
    // only masks with mask_q_seq > 1 need 5D expansion and logits split/fuse.
    const bool gqa_fold_needs_5d_mask =
        gqa_fold && mask_tensor.dims->data[2] != 1;
    if (gqa_batch || gqa_fold_needs_5d_mask) {
      const int32_t expand_axis = 2;
      uint32_t mask_5d_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
          subgraph, /*num_new_axes=*/1, &expand_axis, mask_to_add_id,
          &mask_5d_id, 0));
      mask_to_add_id = mask_5d_id;
    }
    if (gqa_fold_needs_5d_mask) {
      const size_t logits_splits[2] = {g_heads_per_kv, 0};
      uint32_t logits_5d_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_split_dim(subgraph, /*axis=*/2, /*num_splits=*/2,
                               logits_splits, logits_id, &logits_5d_id, 0));

      uint32_t masked_logits_5d_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add,
                                                  logits_5d_id, mask_to_add_id,
                                                  &masked_logits_5d_id, 0));

      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_fuse_dim(subgraph, /*axis=*/2, /*axes_count=*/2,
                              masked_logits_5d_id, &masked_logits_id, 0));
    } else {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add,
                                                  logits_id, mask_to_add_id,
                                                  &masked_logits_id, 0));
    }
  } else {
    masked_logits_id = logits_id;
  }

  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  if (is_float_pv_dot) {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
        subgraph, GetYnnType(pv_dot_type), 0, nullptr, nullptr, 0, &probs_id));
  }
  TF_LITE_ENSURE_YNN_STATUS(
      ynn::define_softmax(subgraph, masked_logits_id, 1.0f, probs_id));

  if (sdpa_inputs.param_index != -1) {
    // Slice V along its kv-seq axis against a value whose matching slinky dim
    // carries the K slice's symbolic extent (slice_like matches dims by
    // position, so the template must have the same layout along that axis).
    // Seq-major V [B, S, H, D] shares K's layout, so the sliced K works;
    // transposed V [B, H, D, S] has kv-seq last, like the scores.
    uint32_t v_template_id = is_seq_major ? current_k_val_id : scores_id;
    int32_t slice_axes_v[1] = {v_seq_axis};
    uint32_t sliced_v_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
        subgraph, /*num_axes=*/1, slice_axes_v, current_v_val_id, v_template_id,
        &sliced_v_val_id, /*flags=*/0));
    current_v_val_id = sliced_v_val_id;
  }
  if (v_tensor.type == kTfLiteInt4) {
    uint32_t v_int8_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(subgraph, current_v_val_id,
                                                 ynn_type_int8, &v_int8_id, 0));
    current_v_val_id = v_int8_id;
  }
  if (gqa_batch) {
    const int32_t expand_axis = 2;
    uint32_t v_5d_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, /*num_new_axes=*/1, &expand_axis, current_v_val_id, &v_5d_id,
        0));
    current_v_val_id = v_5d_id;
  }

  // O = P @ V.
  // When `is_small_q`, compute O = (V @ P^T)^T to avoid transposing V.
  uint32_t bmm2_b_id = current_v_val_id;
  if (!is_small_q) {
    // Bring V to [B, H, S, D].
    const int32_t seq_major_v_perm[] = {2, 1};
    uint32_t v_trans_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 2, is_seq_major ? seq_major_v_perm : swap_last_two_perm,
        current_v_val_id, &v_trans_id, YNN_NODE_FLAG_KEEP_DIMS));
    bmm2_b_id = v_trans_id;
  }
  const TfLiteTensor* output_quant_tensor =
      IsQuantized(output_tensor) ? &output_tensor : nullptr;
  TF_LITE_ENSURE_STATUS(DefineSdpaDot(
      context, subgraph, GetYnnType(pv_dot_type), probs_id,
      /*a_is_quantized=*/false, YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
      bmm2_b_id, v_tensor, output_quant_tensor, GetYnnType(output_tensor.type),
      post_bmm_ptr, /*transpose_ab=*/is_small_q));

  uint32_t post_trans_id = *post_bmm_ptr;
  uint32_t* post_trans_ptr = &post_trans_id;

  if (g_heads_per_kv > 1) {
    uint32_t out_5d_id = *post_bmm_ptr;
    if (gqa_fold) {
      const size_t out_splits[2] = {g_heads_per_kv, 0};
      out_5d_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_split_dim(subgraph, /*axis=*/2, /*num_splits=*/2,
                               out_splits, *post_bmm_ptr, &out_5d_id, 0));
    }
    post_trans_ptr = need_slice_out ? &post_trans_id : &output_val_id;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
        subgraph, /*axis=*/1, /*axes_count=*/2, out_5d_id, post_trans_ptr, 0));
  } else if (is_seq_major) {
    if (!need_slice_out) {
      post_trans_ptr = &output_val_id;
    } else {
      post_trans_ptr = &post_trans_id;
    }
    const int32_t io_perm[] = {0, 2, 1, 3};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, *post_bmm_ptr, post_trans_ptr, 0));
  }

  if (need_slice_out) {
    size_t full_dims_out[YNN_MAX_TENSOR_RANK];
    for (int i = 0; i < output_tensor.dims->size; ++i) {
      full_dims_out[i] = output_tensor.dims->data[i];
    }
    uint32_t dummy_val_id_out = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
        context, subgraph, next_external_id, dummy_inputs,
        sdpa_inputs.param_index, q_seq_axis, output_tensor.dims->size,
        full_dims_out, GetYnnType(output_tensor.type), &dummy_val_id_out));

    int32_t slice_axes_out[1] = {q_seq_axis};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
        subgraph, /*num_axes=*/1, slice_axes_out, *post_trans_ptr,
        dummy_val_id_out, &output_val_id, YNN_NODE_FLAG_KEEP_SHAPE));
  }

  tensor_to_value_id[node.outputs[0]] = output_val_id;
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
