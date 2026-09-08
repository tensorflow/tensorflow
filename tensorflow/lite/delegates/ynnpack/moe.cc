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

bool IsMoe(TfLiteContext* context, int node_index) {
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  if (context->GetNodeAndRegistration(context, node_index, &node,
                                      &registration) != kTfLiteOk) {
    return false;
  }
  return IsMoe(registration, node);
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
  if (output.dims != nullptr && output.dims->size > 0) {
    TF_LITE_ENSURE_EQ(context, output.dims->size, 3);
  }

  TF_LITE_ENSURE(context, tflite::NumElements(&scale) >= 1);

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

  uint32_t gate_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[3]);
  uint32_t up_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[4]);
  uint32_t down_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[5]);
  uint32_t scale_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[6]);

  int D_in = gate_weights.dims->data[3];
  int D_out = D_in;
  int K = expert_indices.dims->data[2];

  // 1. Fuse unit axis 2 into expert dimension E:
  // [*, E, 1, *] -> [*, E, *]
  // and transpose statically to [E, *, *]
  // so expert dimension E is on axis 0 and can be gathered directly.
  uint32_t w_gate_3d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
      subgraph, /*axis=*/1, /*axes_count=*/2, gate_weights_val, &w_gate_3d, 0));

  int32_t perm_static[] = {1, 2, 0};
  uint32_t w_gate_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_static, w_gate_3d, &w_gate_transposed, 0));

  uint32_t w_up_3d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
      subgraph, /*axis=*/1, /*axes_count=*/2, up_weights_val, &w_up_3d, 0));

  uint32_t w_up_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_static, w_up_3d, &w_up_transposed, 0));

  uint32_t w_down_3d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
      subgraph, /*axis=*/1, /*axes_count=*/2, down_weights_val, &w_down_3d, 0));

  uint32_t w_down_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_static, w_down_3d, &w_down_transposed, 0));

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

  // 2. Reshape tokens: [B, N, D_in] -> [M, 1, D_in] where M = B * N
  uint32_t tokens_3d = YNN_INVALID_VALUE_ID;
  size_t shape_M_1_Din[] = {0, 1, static_cast<size_t>(D_in)};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 3, shape_M_1_Din, tokens_val, &tokens_3d, 0));

  // 3. Transpose and reshape expert indices:
  // [B, N, K] -> [K, B, N] -> [K, M, 1, 1]
  int32_t perm_K_B_N[] = {2, 0, 1};
  uint32_t ei_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_K_B_N, expert_indices_val, &ei_transposed, 0));

  uint32_t ei_4d = YNN_INVALID_VALUE_ID;
  size_t shape_K_M_1_1[] = {static_cast<size_t>(K), 0, 1, 1};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 4, shape_K_M_1_1, ei_transposed, &ei_4d, 0));

  // 4. Gather weights along axis 0 (expert dim E) for active choices:
  // [E, D_in, D_mid] gathered by [K, M, 1, 1] -> [K, M, D_in, D_mid]
  int32_t gather_axis_0 = 0;
  uint32_t w_gate_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 4, w_gate_transposed, ei_4d, &w_gate_k, 0));

  uint32_t w_up_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 4, w_up_transposed, ei_4d, &w_up_k, 0));

  uint32_t w_down_k = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &gather_axis_0, 4, w_down_transposed, ei_4d, &w_down_k, 0));

  // 5. Batched dot products:
  // tokens [M, 1, D_in] @ w_gate_k [K, M, D_in, D_mid] -> gate [K, M, 1, D_mid]
  // Slinky automatically broadcasts the trailing K dimension of tokens.
  uint32_t gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_3d, w_gate_k,
                                           YNN_INVALID_VALUE_ID, &gate, 0));

  uint32_t gelu_gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_approx_gelu(subgraph, gate, gelu_gate));

  uint32_t up = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_3d, w_up_k,
                                           YNN_INVALID_VALUE_ID, &up, 0));

  uint32_t mid = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_multiply, gelu_gate, up, &mid, 0));

  // mid [K, M, 1, D_mid] @ w_down_k [K, M, D_mid, D_out] ->
  //   out_4d [K, M, 1, D_out]
  uint32_t out_4d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, mid, w_down_k,
                                           YNN_INVALID_VALUE_ID, &out_4d, 0));

  // 6. Routing weights & scale: [B, N, K] -> [K, B, N] -> [K, M, 1, 1]
  uint32_t rw_transposed = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 3, perm_K_B_N, routing_weights_val, &rw_transposed, 0));

  uint32_t rw_4d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 4, shape_K_M_1_1, rw_transposed, &rw_4d, 0));

  uint32_t rw_scaled = rw_4d;
  size_t num_scale_elements = tflite::NumElements(&scale);
  if (num_scale_elements == 1) {
    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_multiply, rw_4d, scale_val, &rw_scaled, 0));
  } else if (num_scale_elements > 1) {
    uint32_t scale_1d = YNN_INVALID_VALUE_ID;
    size_t shape_E[] = {num_scale_elements};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
        subgraph, 1, shape_E, scale_val, &scale_1d, 0));

    uint32_t scale_gathered = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 3,
                                                scale_1d, ei_transposed,
                                                &scale_gathered, 0));

    uint32_t scale_4d = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
        subgraph, 4, shape_K_M_1_1, scale_gathered, &scale_4d, 0));

    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_multiply, rw_4d, scale_4d, &rw_scaled, 0));
  }

  // 7. Multiply weighted outputs:
  // [K, M, 1, D_out] * [K, M, 1, 1] -> [K, M, 1, D_out]
  uint32_t weighted_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
      subgraph, ynn_binary_multiply, out_4d, rw_scaled, &weighted_out, 0));

  // 8. Reduce sum over K (axis 0): [K, M, 1, D_out] -> [M, 1, D_out]
  uint32_t reduced_out = YNN_INVALID_VALUE_ID;
  int32_t reduce_axis_0 = 0;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &reduce_axis_0,
                        weighted_out, YNN_INVALID_VALUE_ID, &reduced_out, 0));

  // 9. Reshape to final output: [M, 1, D_out] -> [B, N, D_out]
  size_t b_dim =
      (tokens.dims && tokens.dims->size >= 1 && tokens.dims->data[0] > 0)
          ? static_cast<size_t>(tokens.dims->data[0])
          : 1;
  size_t shape_B_N_Dout[] = {b_dim, 0, static_cast<size_t>(D_out)};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 3, shape_B_N_Dout, reduced_out, &output_val, 0));

  tensor_to_value_id[node.outputs[0]] = output_val;

  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
