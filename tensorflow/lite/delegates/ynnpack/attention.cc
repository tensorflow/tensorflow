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
      auto is_float_or_bool = [](TfLiteType type) {
        return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
               type == kTfLiteBFloat16 || type == kTfLiteBool;
      };
      if (is_float_or_bool(tensor3.type)) {
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

bool IsSdpa(TfLiteContext* context, int node_index) {
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  if (context == nullptr ||
      context->GetNodeAndRegistration(context, node_index, &node,
                                      &registration) != kTfLiteOk) {
    return false;
  }
  return IsSdpa(registration, node);
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

  auto is_float_type = [](TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
           type == kTfLiteBFloat16;
  };

  TF_LITE_ENSURE(context, IsTensorSupported(q));
  TF_LITE_ENSURE(context, is_float_type(q.type));
  TF_LITE_ENSURE(context, IsTensorSupported(k));
  TF_LITE_ENSURE(context, is_float_type(k.type));
  TF_LITE_ENSURE(context, IsTensorSupported(v));
  TF_LITE_ENSURE(context, is_float_type(v.type));
  TF_LITE_ENSURE(context, IsTensorSupported(output));
  TF_LITE_ENSURE(context, is_float_type(output.type));

  TF_LITE_ENSURE_EQ(context, q.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, k.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, v.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output.dims->size, 4);

  // If 4th input is present, it can be Mask or Param.
  if (node->inputs->size >= 4 && node->inputs->data[3] != -1) {
    const TfLiteTensor& input3 = context->tensors[node->inputs->data[3]];
    TF_LITE_ENSURE(context,
                   input3.type == kTfLiteBool || IsTensorSupported(input3));
    if (is_float_type(input3.type) || input3.type == kTfLiteBool) {
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

  TfLiteNode* tflite_node = nullptr;
  TfLiteRegistration* reg = nullptr;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));

  bool is_seq_major = true;
  if (reg && reg->builtin_code == kTfLiteBuiltinStablehloComposite) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node->builtin_data);
    if (composite_params != nullptr && composite_params->name != nullptr) {
      if (strcmp(composite_params->name, "odml.sdpa_transposed") == 0) {
        is_seq_major = false;
      }
    }
  } else if (reg && reg->builtin_code == kTfLiteBuiltinCustom &&
             reg->custom_name != nullptr) {
    if (strcmp(reg->custom_name, "odml.sdpa_transposed") == 0) {
      is_seq_major = false;
    }
  }

  float scale_val = 1.0f;
  bool scale_specified = false;
  const flexbuffers::Map flexbuffer_map = GetFlexBufferMap(reg, tflite_node);
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

  uint32_t current_k_val_id = k_val_id;
  uint32_t current_v_val_id = v_val_id;
  uint32_t current_mask_val_id = mask_val_id;

  if (sdpa_inputs.param_index != -1) {
    // Slice K
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

    // Slice V
    const TfLiteTensor& v_tensor = context->tensors[sdpa_inputs.v_index];
    size_t full_dims_v[YNN_MAX_TENSOR_RANK];
    for (int i = 0; i < v_tensor.dims->size; ++i) {
      full_dims_v[i] = v_tensor.dims->data[i];
    }
    uint32_t dummy_val_id_v = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
        context, subgraph, next_external_id, dummy_inputs,
        sdpa_inputs.param_index, v_seq_axis, v_tensor.dims->size, full_dims_v,
        GetYnnType(v_tensor.type), &dummy_val_id_v));

    uint32_t sliced_v_val_id = YNN_INVALID_VALUE_ID;
    int32_t slice_axes_v[1] = {v_seq_axis};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
        subgraph, /*num_axes=*/1, slice_axes_v, current_v_val_id,
        dummy_val_id_v, &sliced_v_val_id, /*flags=*/0));
    current_v_val_id = sliced_v_val_id;

    if (mask_val_id != YNN_INVALID_VALUE_ID) {
      const TfLiteTensor& mask_tensor =
          context->tensors[sdpa_inputs.mask_index];
      size_t full_dims_mask[YNN_MAX_TENSOR_RANK];
      for (int i = 0; i < mask_tensor.dims->size; ++i) {
        full_dims_mask[i] = mask_tensor.dims->data[i];
      }
      int mask_seq_axis = mask_tensor.dims->size - 1;
      uint32_t dummy_val_id_mask = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(GetOrCreateDummyInput(
          context, subgraph, next_external_id, dummy_inputs,
          sdpa_inputs.param_index, mask_seq_axis, mask_tensor.dims->size,
          full_dims_mask, GetYnnType(mask_tensor.type), &dummy_val_id_mask));

      uint32_t sliced_mask_val_id = YNN_INVALID_VALUE_ID;
      int32_t slice_axes_mask[1] = {mask_seq_axis};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, /*num_axes=*/1, slice_axes_mask, current_mask_val_id,
          dummy_val_id_mask, &sliced_mask_val_id, /*flags=*/0));
      current_mask_val_id = sliced_mask_val_id;
    }
  }

  uint32_t mask_val_to_add_id = current_mask_val_id;
  if (sdpa_inputs.mask_index != -1) {
    const TfLiteTensor& mask_tensor = context->tensors[sdpa_inputs.mask_index];
    if (mask_tensor.type == kTfLiteBool) {
      uint32_t mask_float_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_convert(
          subgraph, current_mask_val_id, ynn_type_fp32, &mask_float_id, 0));

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

      mask_val_to_add_id = float_mask_id;
    }
  }

  uint32_t q_trans_id = YNN_INVALID_VALUE_ID;
  uint32_t k_trans_id = YNN_INVALID_VALUE_ID;
  uint32_t v_trans_id = YNN_INVALID_VALUE_ID;

  if (is_seq_major) {
    const int32_t io_perm[] = {0, 2, 1, 3};

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, q_val_id, &q_trans_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, current_k_val_id, &k_trans_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, io_perm, current_v_val_id, &v_trans_id, 0));
  } else {
    // For sdpa_transposed, Q and K are already in correct layout.
    q_trans_id = q_val_id;
    k_trans_id = current_k_val_id;

    // V is transposed [B, H, D, S], we need to transpose it to [B, H, S, D].
    const int32_t v_perm[] = {0, 1, 3, 2};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, v_perm, current_v_val_id, &v_trans_id, 0));
  }

  uint32_t scale_const_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &scale_val,
                        YNN_VALUE_FLAG_COPY_DATA_FP32, &scale_const_id));

  const int q_seq_dim = is_seq_major ? 1 : 2;
  bool use_decode1 = (q_tensor.dims->data[q_seq_dim] <= 32);

  bool need_slice_out = false;
  uint32_t post_bmm_id = YNN_INVALID_VALUE_ID;
  uint32_t* post_bmm_ptr = &post_bmm_id;

  if (!need_slice_out && !is_seq_major) {
    post_bmm_ptr = &output_val_id;
  }

  if (use_decode1) {
    uint32_t q_scaled_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                q_trans_id, scale_const_id,
                                                &q_scaled_id, 0));

    uint32_t q_scaled_t_id = YNN_INVALID_VALUE_ID;
    const int32_t q_t_perm[] = {0, 1, 3, 2};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, q_t_perm, q_scaled_id, &q_scaled_t_id, 0));

    uint32_t scores_ts_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dot(subgraph, /*num_k_dims=*/1, k_trans_id, q_scaled_t_id,
                       YNN_INVALID_VALUE_ID, &scores_ts_id, 0));

    uint32_t scores_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, q_t_perm, scores_ts_id, &scores_id, 0));

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
    if (mask_val_to_add_id != YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add,
                                                  logits_id, mask_val_to_add_id,
                                                  &masked_logits_id, 0));
    } else {
      masked_logits_id = logits_id;
    }

    uint32_t probs_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn::define_softmax(subgraph, masked_logits_id, 1.0f, probs_id));

    if (!is_seq_major) {
      // Rewrite BMM2: O = (V @ P^T)^T to avoid transposing V.
      // P is [B, N, 1, S] -> P^T is [B, N, S, 1]
      // V is [B, N, H, S]
      // V @ P^T is [B, N, H, 1] -> transpose to [B, N, 1, H]
      uint32_t probs_t_id = YNN_INVALID_VALUE_ID;
      const int32_t probs_t_perm[] = {0, 1, 3, 2};
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
          subgraph, 4, probs_t_perm, probs_id, &probs_t_id, 0));

      uint32_t post_bmm_t_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_dot(subgraph, /*num_k_dims=*/1, current_v_val_id,
                         probs_t_id, YNN_INVALID_VALUE_ID, &post_bmm_t_id, 0));

      TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
          subgraph, 4, probs_t_perm, post_bmm_t_id, post_bmm_ptr, 0));
    } else {
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, v_trans_id,
                         YNN_INVALID_VALUE_ID, post_bmm_ptr, 0));
    }

  } else {
    // General case: S = Q @ K^T
    uint32_t q_scaled_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                q_trans_id, scale_const_id,
                                                &q_scaled_id, 0));

    uint32_t k_trans_t_id = YNN_INVALID_VALUE_ID;
    const int32_t k_t_perm[] = {0, 1, 3, 2};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 4, k_t_perm, k_trans_id, &k_trans_t_id, 0));

    uint32_t scores_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dot(subgraph, /*num_k_dims=*/1, q_scaled_id, k_trans_t_id,
                       YNN_INVALID_VALUE_ID, &scores_id, 0));

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
    if (mask_val_to_add_id != YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add,
                                                  logits_id, mask_val_to_add_id,
                                                  &masked_logits_id, 0));
    } else {
      masked_logits_id = logits_id;
    }

    uint32_t probs_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn::define_softmax(subgraph, masked_logits_id, 1.0f, probs_id));

    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, v_trans_id,
                       YNN_INVALID_VALUE_ID, post_bmm_ptr, 0));
  }

  uint32_t post_trans_id = *post_bmm_ptr;
  uint32_t* post_trans_ptr = &post_trans_id;

  if (is_seq_major) {
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
