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

#include "tensorflow/lite/delegates/ynnpack/moe.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {

bool IsMoe(const TfLiteRegistration* registration, const TfLiteNode* node) {
  if (registration == nullptr) {
    return false;
  }
  if (registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node != nullptr && node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    return composite_params->name != nullptr &&
           strcmp(composite_params->name, "odml.moe_experts") == 0;
  }
  return false;
}

TfLiteStatus IsMoeSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE(context, IsMoe(registration, node));
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 7);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& tokens = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& routing_weights = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& expert_indices = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& gate_weights = context->tensors[node->inputs->data[3]];
  const TfLiteTensor& up_weights = context->tensors[node->inputs->data[4]];
  const TfLiteTensor& down_weights = context->tensors[node->inputs->data[5]];
  const TfLiteTensor& scale = context->tensors[node->inputs->data[6]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, IsTensorSupported(tokens));
  TF_LITE_ENSURE(context, IsTensorSupported(routing_weights));
  TF_LITE_ENSURE(context, IsTensorSupported(expert_indices));
  TF_LITE_ENSURE(context, IsTensorSupported(gate_weights));
  TF_LITE_ENSURE(context, IsTensorSupported(up_weights));
  TF_LITE_ENSURE(context, IsTensorSupported(down_weights));
  TF_LITE_ENSURE(context, IsTensorSupported(scale));
  TF_LITE_ENSURE(context, IsTensorSupported(output));

  TF_LITE_ENSURE(context, tflite::IsConstantTensor(&gate_weights));
  TF_LITE_ENSURE(context, tflite::IsConstantTensor(&up_weights));
  TF_LITE_ENSURE(context, tflite::IsConstantTensor(&down_weights));
  TF_LITE_ENSURE(context, tflite::IsConstantTensor(&scale));

  TF_LITE_ENSURE_EQ(context, tokens.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, routing_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, expert_indices.type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, gate_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, up_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, down_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, scale.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output.type, kTfLiteFloat32);

  TF_LITE_ENSURE(context, tokens.dims != nullptr && tokens.dims->size == 3);
  TF_LITE_ENSURE(context, routing_weights.dims != nullptr &&
                              routing_weights.dims->size == 3);
  TF_LITE_ENSURE(context, expert_indices.dims != nullptr &&
                              expert_indices.dims->size == 3);
  TF_LITE_ENSURE(context,
                 gate_weights.dims != nullptr && gate_weights.dims->size == 4);
  TF_LITE_ENSURE(context,
                 up_weights.dims != nullptr && up_weights.dims->size == 4);
  TF_LITE_ENSURE(context,
                 down_weights.dims != nullptr && down_weights.dims->size == 4);
  TF_LITE_ENSURE(context, output.dims != nullptr && output.dims->size == 3);

  return kTfLiteOk;
}

TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 7);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  const TfLiteTensor& tokens = context->tensors[node.inputs[0]];
  const TfLiteTensor& routing_weights = context->tensors[node.inputs[1]];
  const TfLiteTensor& expert_indices = context->tensors[node.inputs[2]];
  const TfLiteTensor& gate_weights = context->tensors[node.inputs[3]];
  const TfLiteTensor& up_weights = context->tensors[node.inputs[4]];
  const TfLiteTensor& down_weights = context->tensors[node.inputs[5]];
  const TfLiteTensor& scale = context->tensors[node.inputs[6]];

  TF_LITE_ENSURE_EQ(context, tokens.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, routing_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, expert_indices.type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, gate_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, up_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, down_weights.type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, scale.type, kTfLiteFloat32);

  int E = gate_weights.dims->data[1];
  int K = (expert_indices.dims && expert_indices.dims->size >= 1)
              ? expert_indices.dims->data[expert_indices.dims->size - 1]
              : 0;
  TF_LITE_ENSURE(context, K > 0);

  TF_LITE_ENSURE(context, tflite::NumElements(&scale) == 1 ||
                              tflite::NumElements(&scale) == E);

  uint32_t gate_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[3]);
  uint32_t up_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[4]);
  uint32_t down_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[5]);
  uint32_t scale_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[6]);

  // 1. Transpose constant weights from 4D [*, E, 1, *] to 3D [E, *, *]
  // dropping the unit axis 2 so expert dimension E is on axis 0 and can be
  // gathered directly.
  int32_t perm_weights[] = {1, 3, 0};
  uint32_t w_gate_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_weights, gate_weights_val, &w_gate_transposed, 0));

  uint32_t w_up_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_weights, up_weights_val, &w_up_transposed, 0));

  uint32_t w_down_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_weights, down_weights_val, &w_down_transposed, 0));

  uint32_t tokens_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[0]);
  uint32_t routing_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[1]);
  uint32_t expert_indices_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[2]);
  uint32_t output_val = YNN_INVALID_VALUE_ID;
  auto out_it = tensor_to_value_id.find(node.outputs[0]);
  if (out_it != tensor_to_value_id.end()) {
    output_val = out_it->second;
  }

  // 2. Expand tokens: [B, N, D_in] -> [B, N, 1, 1, D_in]
  uint32_t tokens_5d = YNN_INVALID_VALUE_ID;
  int32_t expand_axes_2_3[] = {2, 3};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 2, expand_axes_2_3, tokens_val, &tokens_5d, 0));

  // 3. Expand expert indices: [B, N, K] -> [B, N, K, 1, 1]
  uint32_t ei_5d = YNN_INVALID_VALUE_ID;
  int32_t expand_axes_3_4[] = {3, 4};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 2, expand_axes_3_4, expert_indices_val, &ei_5d, 0));

  // 4. Gather weights along axis 0 (expert dim E) for active choices:
  // [E, D_in, D_mid] gathered by [B, N, K, 1, 1] -> [B, N, K, D_in, D_mid]
  int32_t gather_axis_0 = 0;
  uint32_t w_gate_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 5, w_gate_transposed, ei_5d, &w_gate_k, 0));

  uint32_t w_up_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 5, w_up_transposed, ei_5d, &w_up_k, 0));

  uint32_t w_down_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 5, w_down_transposed, ei_5d, &w_down_k, 0));

  // 5. Batched dot products:
  // tokens [B, N, 1, 1, D_in] @ w_gate_k [B, N, K, D_in, D_mid] -> gate [B, N,
  // K, 1, D_mid] Slinky automatically broadcasts axis 2 (dim 1 -> K) of tokens.
  uint32_t gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_5d, w_gate_k,
                                           YNN_INVALID_VALUE_ID, &gate, 0));

  uint32_t gelu_gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_approx_gelu(subgraph, gate, gelu_gate));

  uint32_t up = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_5d, w_up_k,
                                           YNN_INVALID_VALUE_ID, &up, 0));

  uint32_t mid = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_multiply, gelu_gate, up, &mid, 0));

  // mid [B, N, K, 1, D_mid] @ w_down_k [B, N, K, D_mid, D_out] -> out_5d [B, N,
  // K, 1, D_out]
  uint32_t out_5d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, mid, w_down_k,
                                           YNN_INVALID_VALUE_ID, &out_5d, 0));

  // 6. Squeeze axis 3: [B, N, K, 1, D_out] -> [B, N, K, D_out]
  // TODO: b/558433816 - This should be static_transpose, but it is faster to
  // use a fuse_dim due to scheduling issues.
  uint32_t out_4d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_fuse_dim(subgraph, 2, 2, out_5d, &out_4d, 0));

  // 7. Routing weights & scale: [B, N, K] -> [B, N, K, 1]
  uint32_t scale_1d = YNN_INVALID_VALUE_ID;
  if (scale.dims != nullptr && scale.dims->size > 1) {
    int32_t perm_last_axis[] = {scale.dims->size - 1};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 1, perm_last_axis, scale_val, &scale_1d, 0));
  } else {
    scale_1d = scale_val;
  }

  uint32_t rw_scaled = routing_weights_val;
  size_t num_scale_elements = tflite::NumElements(&scale);
  if (num_scale_elements == 1) {
    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                routing_weights_val, scale_1d,
                                                &rw_scaled, 0));
  } else if (num_scale_elements == static_cast<size_t>(E)) {
    uint32_t scale_gathered = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 3,
                                                scale_1d, expert_indices_val,
                                                &scale_gathered, 0));

    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                routing_weights_val,
                                                scale_gathered, &rw_scaled, 0));
  }

  uint32_t rw_4d = YNN_INVALID_VALUE_ID;
  int32_t expand_axis_3 = 3;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 1, &expand_axis_3, rw_scaled, &rw_4d, 0));

  // 8. Multiply weighted outputs: [B, N, K, D_out] * [B, N, K, 1] -> [B, N, K,
  // D_out]
  uint32_t weighted_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                              out_4d, rw_4d, &weighted_out, 0));

  // 9. Reduce sum over K (axis 2): [B, N, K, D_out] -> [B, N, D_out]
  int32_t reduce_axis_2 = 2;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &reduce_axis_2,
                        weighted_out, YNN_INVALID_VALUE_ID, &output_val, 0));

  tensor_to_value_id[node.outputs[0]] = output_val;

  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
