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
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

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
      if (IsFloatAttentionType(tensor3.type) || tensor3.type == kTfLiteBool) {
        inputs.mask_index = idx3;
      } else {
        inputs.param_index = idx3;
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

// Computes `output = a @ b` where `a` is float [..., M, K] and `b` is float or
// per-tensor quantized int8 [..., K, N]. When `b` is quantized, `a` is
// dynamically quantized to int8 along its reduction axis (-1).
TfLiteStatus DefineSdpaDot(TfLiteContext* context, ynn_subgraph_t subgraph,
                           uint32_t a_id, uint32_t b_id,
                           const TfLiteTensor& b_tensor, ynn_type output_type,
                           uint32_t* output_id) {
  if (!IsQuantized(b_tensor)) {
    if (*output_id == YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
          subgraph, output_type, 0, nullptr, nullptr, 0, output_id));
    }
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_id,
                                             b_id, YNN_INVALID_VALUE_ID,
                                             output_id, 0));
    return kTfLiteOk;
  }

  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(context, subgraph, b_tensor,
                                                 &b_scale_id, &b_zp_id));

  int32_t reduce_axis = -1;
  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
      subgraph, ynn_reduce_min_max, 1, &reduce_axis, a_id, YNN_INVALID_VALUE_ID,
      &min_max_id, YNN_NODE_FLAG_KEEP_DIMS));

  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dynamic_quantization(
      subgraph, min_max_id, ynn_type_int8, &a_zp_id, &a_scale_id, 0));

  uint32_t q_a_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(
      subgraph, a_id, ynn_type_int8, a_zp_id, a_scale_id, &q_a_id, 0));

  uint32_t dot_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_dot_quantization(
      subgraph, /*num_k_dims=*/1, q_a_id, a_zp_id, a_scale_id, b_id, b_zp_id,
      b_scale_id, dot_zp_id, dot_scale_id));

  uint32_t accum_init_id = YNN_INVALID_VALUE_ID;
  if (dot_zp_id != YNN_INVALID_VALUE_ID) {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(subgraph, ynn_unary_negate,
                                               dot_zp_id, &accum_init_id, 0));
  }

  uint32_t accum_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, q_a_id,
                                           b_id, accum_init_id, &accum_id, 0));

  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_dequantize(subgraph, accum_id, YNN_INVALID_VALUE_ID,
                            dot_scale_id, output_type, output_id, 0));
  return kTfLiteOk;
}

}  // namespace

bool IsFloatAttentionType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
         type == kTfLiteBFloat16;
}

TfLiteStatus DefineAttentionScores(TfLiteContext* context,
                                   ynn_subgraph_t subgraph, uint32_t q_id,
                                   uint32_t k_id, const TfLiteTensor& k_tensor,
                                   bool use_decode_transpose,
                                   uint32_t* scores_id) {
  const int32_t swap_last_two_perm[] = {-1, -2};
  *scores_id = YNN_INVALID_VALUE_ID;
  if (use_decode_transpose) {
    // Compute S^T = K @ Q^T and transpose the (small) result.
    uint32_t q_t_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 2, swap_last_two_perm, q_id,
                                    &q_t_id, YNN_NODE_FLAG_KEEP_DIMS));
    uint32_t scores_ts_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, k_id,
                                             q_t_id, YNN_INVALID_VALUE_ID,
                                             &scores_ts_id, 0));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 2, swap_last_two_perm, scores_ts_id, scores_id,
        YNN_NODE_FLAG_KEEP_DIMS));
  } else {
    uint32_t k_t_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 2, swap_last_two_perm, k_id,
                                    &k_t_id, YNN_NODE_FLAG_KEEP_DIMS));
    TF_LITE_ENSURE_STATUS(DefineSdpaDot(context, subgraph, q_id, k_t_id,
                                        k_tensor, ynn_type_fp32, scores_id));
  }
  return kTfLiteOk;
}

TfLiteStatus ApplyLogitSoftcap(TfLiteContext* context, ynn_subgraph_t subgraph,
                               float cap_val, uint32_t* logits_id) {
  if (cap_val <= 0.0f) return kTfLiteOk;
  uint32_t cap_const_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineScalarConstant(context, subgraph, ynn_type_fp32,
                                             cap_val, &cap_const_id));

  uint32_t scores_div_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_divide,
                                              *logits_id, cap_const_id,
                                              &scores_div_id, 0));

  uint32_t scores_tanh_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(
      subgraph, ynn_unary_tanh, scores_div_id, &scores_tanh_id, 0));

  uint32_t scores_capped_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                              scores_tanh_id, cap_const_id,
                                              &scores_capped_id, 0));
  *logits_id = scores_capped_id;
  return kTfLiteOk;
}

TfLiteStatus DefineAttentionWeightedValues(
    TfLiteContext* context, ynn_subgraph_t subgraph, uint32_t probs_id,
    uint32_t v_id, const TfLiteTensor& v_tensor, ynn_type output_type,
    bool use_decode_transpose, const int32_t v_perm[2], uint32_t* out_id) {
  const int32_t swap_last_two_perm[] = {-1, -2};
  if (use_decode_transpose) {
    // Rewrite BMM2: O = (V @ P^T)^T to avoid transposing V.
    uint32_t probs_t_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 2, swap_last_two_perm, probs_id,
                                    &probs_t_id, YNN_NODE_FLAG_KEEP_DIMS));

    uint32_t post_bmm_t_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
        subgraph, output_type, 0, nullptr, nullptr, 0, &post_bmm_t_id));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, v_id,
                                             probs_t_id, YNN_INVALID_VALUE_ID,
                                             &post_bmm_t_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 2, swap_last_two_perm, post_bmm_t_id, out_id,
        YNN_NODE_FLAG_KEEP_DIMS));
  } else {
    uint32_t v_trans_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 2, v_perm, v_id, &v_trans_id, YNN_NODE_FLAG_KEEP_DIMS));

    TF_LITE_ENSURE_STATUS(DefineSdpaDot(context, subgraph, probs_id, v_trans_id,
                                        v_tensor, output_type, out_id));
  }
  return kTfLiteOk;
}

bool IsSdpa(const TfLiteRegistration* registration, const TfLiteNode* node) {
  return IsCustomOrCompositeOpNamed(registration, node,
                                    "odml.scaled_dot_product_attention") ||
         IsCustomOrCompositeOpNamed(registration, node, "odml.sdpa_transposed");
}

TfLiteStatus IsSdpaSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE(context, IsSdpa(registration, node));
  TF_LITE_ENSURE(context, node->inputs->size >= 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& q = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& k = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& v = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  auto is_quantized_int8 = [](const TfLiteTensor& tensor) {
    return tensor.type == kTfLiteInt8 && IsQuantized(tensor);
  };

  TF_LITE_ENSURE(context, IsTensorSupported(q));
  TF_LITE_ENSURE(context, IsFloatAttentionType(q.type));
  TF_LITE_ENSURE(context, IsTensorSupported(k));
  TF_LITE_ENSURE(context, k.type == q.type || is_quantized_int8(k));
  TF_LITE_ENSURE(context, IsTensorSupported(v));
  TF_LITE_ENSURE(context, IsFloatAttentionType(v.type) || is_quantized_int8(v));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  TF_LITE_ENSURE(context, IsFloatAttentionType(output.type));

  TF_LITE_ENSURE_EQ(context, q.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, k.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, v.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output.dims->size, 4);

  const bool is_seq_major =
      !IsCustomOrCompositeOpNamed(registration, node, "odml.sdpa_transposed");
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
    if (IsFloatAttentionType(input3.type) || input3.type == kTfLiteBool) {
      // If it is Mask, it should be float or bool.
    } else if (input3.type == kTfLiteInt32 || input3.type == kTfLiteInt64) {
      // If it is Param, it should be int32 or int64.
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
  }

  return kTfLiteOk;
}

TfLiteStatus DefineSdpaNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            uint32_t& next_external_id,
                            std::vector<DummyInputInfo>& dummy_inputs,
                            const NodeInfo& node) {
  SdpaInputs sdpa_inputs = GetSdpaInputs(context, node);

  const TfLiteTensor& q_tensor = context->tensors[sdpa_inputs.q_index];
  const TfLiteTensor& k_tensor = context->tensors[sdpa_inputs.k_index];
  const TfLiteTensor& v_tensor = context->tensors[sdpa_inputs.v_index];
  const TfLiteTensor& output_tensor = context->tensors[node.outputs[0]];

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
  if (!flexbuffer_map["logit_cap"].IsNull()) {
    logit_cap_val = flexbuffer_map["logit_cap"].AsFloat();
  }

  const int k_seq_axis = is_seq_major ? 1 : 2;
  const int v_seq_axis = is_seq_major ? 1 : 3;
  const int q_seq_axis = is_seq_major ? 1 : 2;

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
        GetYnnType(k_tensor.type), &dummy_val_id_k));

    uint32_t sliced_k_val_id = YNN_INVALID_VALUE_ID;
    int32_t slice_axes_k[1] = {k_seq_axis};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
        subgraph, /*num_axes=*/1, slice_axes_k, current_k_val_id,
        dummy_val_id_k, &sliced_k_val_id, /*flags=*/0));
    current_k_val_id = sliced_k_val_id;
  }

  uint32_t mask_val_to_add_id = current_mask_val_id;
  if (sdpa_inputs.mask_index != -1) {
    const TfLiteTensor& mask_tensor = context->tensors[sdpa_inputs.mask_index];
    if (mask_tensor.type == kTfLiteBool) {
      uint32_t mask_float_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(
          subgraph, current_mask_val_id, ynn_type_fp32, &mask_float_id, 0));

      uint32_t one_const_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(DefineScalarConstant(
          context, subgraph, ynn_type_fp32, 1.0f, &one_const_id));

      uint32_t mask_fill_const_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(
          DefineScalarConstant(context, subgraph, ynn_type_fp32,
                               kAttentionMaskFillValue, &mask_fill_const_id));

      uint32_t inv_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_subtract,
                                                  one_const_id, mask_float_id,
                                                  &inv_mask_id, 0));

      uint32_t float_mask_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_binary(subgraph, ynn_binary_multiply, inv_mask_id,
                            mask_fill_const_id, &float_mask_id, 0));

      mask_val_to_add_id = float_mask_id;
    }
  }

  // Bring Q and K to [B, H, S, D]. V is handled below, next to the dot that
  // consumes it.
  uint32_t q_trans_id = YNN_INVALID_VALUE_ID;
  uint32_t k_trans_id = YNN_INVALID_VALUE_ID;

  if (is_seq_major) {
    const int32_t io_perm[] = {0, 2, 1, 3};

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, q_val_id, &q_trans_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, current_k_val_id, &k_trans_id, 0));
  } else {
    // For sdpa_transposed, Q and K are already in correct layout.
    q_trans_id = q_val_id;
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
  const bool use_decode1 =
      is_small_q && !IsQuantized(k_tensor) && !IsQuantized(v_tensor);
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
  uint32_t q_scaled_id = q_trans_id;
  if (scale_val != 1.0f) {
    uint32_t scale_const_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
        subgraph, GetYnnType(q_tensor.type), 0, nullptr, &scale_val,
        YNN_VALUE_FLAG_COPY_DATA_FP32, &scale_const_id));
    q_scaled_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                q_trans_id, scale_const_id,
                                                &q_scaled_id, 0));
  }

  // Scores: S = Q @ K^T, [B, H, Q, S].
  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineAttentionScores(context, subgraph, q_scaled_id,
                                              k_trans_id, k_tensor, use_decode1,
                                              &scores_id));

  uint32_t logits_id = scores_id;
  TF_LITE_ENSURE_STATUS(
      ApplyLogitSoftcap(context, subgraph, logit_cap_val, &logits_id));

  uint32_t masked_logits_id = YNN_INVALID_VALUE_ID;
  if (mask_val_to_add_id != YNN_INVALID_VALUE_ID) {
    const TfLiteTensor& mask_tensor = context->tensors[sdpa_inputs.mask_index];
    uint32_t mask_to_add_id = mask_val_to_add_id;
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
  if (!IsQuantized(v_tensor)) {
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, GetYnnType(v_tensor.type), 0, nullptr,
                          nullptr, 0, &probs_id));
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
  if (gqa_batch) {
    const int32_t expand_axis = 2;
    uint32_t v_5d_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, /*num_new_axes=*/1, &expand_axis, current_v_val_id, &v_5d_id,
        0));
    current_v_val_id = v_5d_id;
  }

  // O = P @ V.
  const int32_t swap_last_two_perm[] = {-1, -2};
  const int32_t seq_major_v_perm[] = {2, 1};
  TF_LITE_ENSURE_STATUS(DefineAttentionWeightedValues(
      context, subgraph, probs_id, current_v_val_id, v_tensor,
      GetYnnType(output_tensor.type), use_decode1,
      is_seq_major ? seq_major_v_perm : swap_last_two_perm, post_bmm_ptr));

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
