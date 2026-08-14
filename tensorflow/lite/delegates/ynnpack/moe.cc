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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
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

namespace {

void TransposeWeightsGateUp(const float* src, float* dst, int E, int D_mid,
                            int D_in) {
  for (int e = 0; e < E; ++e) {
    for (int d_in = 0; d_in < D_in; ++d_in) {
      for (int d_mid = 0; d_mid < D_mid; ++d_mid) {
        int src_idx = d_mid * E * D_in + e * D_in + d_in;
        int dst_idx = e * D_in * D_mid + d_in * D_mid + d_mid;
        dst[dst_idx] = src[src_idx];
      }
    }
  }
}

void TransposeAndScaleWeightsDown(const float* src, const float* scale,
                                  size_t num_scale_elements, float* dst, int E,
                                  int D_in, int D_mid) {
  for (int e = 0; e < E; ++e) {
    float s = (num_scale_elements == 1) ? scale[0] : scale[e];
    for (int d_mid = 0; d_mid < D_mid; ++d_mid) {
      for (int d_in = 0; d_in < D_in; ++d_in) {
        int src_idx = d_in * E * D_mid + e * D_mid + d_mid;
        int dst_idx = e * D_mid * D_in + d_mid * D_in + d_in;
        dst[dst_idx] = src[src_idx] * s;
      }
    }
  }
}

ynn_status DefineGather2D(ynn_subgraph_t sub, uint32_t input_id,
                          uint32_t index_id, size_t d_in, uint32_t* output_id) {
  uint32_t tokens_3d = YNN_INVALID_VALUE_ID;
  int32_t expand_axis_1 = 1;
  ynn_status status = ynn_define_static_expand_dims(sub, 1, &expand_axis_1,
                                                    input_id, &tokens_3d, 0);
  if (status != ynn_status_success) return status;

  uint32_t index_3d = YNN_INVALID_VALUE_ID;
  int32_t expand_axes_1_2[] = {1, 2};
  status = ynn_define_static_expand_dims(sub, 2, expand_axes_1_2, index_id,
                                         &index_3d, 0);
  if (status != ynn_status_success) return status;

  uint32_t index_broadcast = YNN_INVALID_VALUE_ID;
  int32_t broadcast_axis_2 = 2;
  status = ynn_define_broadcast(sub, 1, &broadcast_axis_2, index_3d,
                                &index_broadcast, 0);
  if (status != ynn_status_success) return status;

  uint32_t gathered_3d = YNN_INVALID_VALUE_ID;
  int32_t axis_0 = 0;
  status = ynn_define_gather(sub, 1, &axis_0, 3, tokens_3d, index_broadcast,
                             &gathered_3d, 0);
  if (status != ynn_status_success) return status;

  size_t shape_M_Din[] = {0, d_in};
  return ynn_define_static_reshape(sub, 2, shape_M_Din, gathered_3d, output_id,
                                   0);
}

}  // namespace

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

  TF_LITE_ENSURE_EQ(context, tokens.dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, routing_weights.dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, expert_indices.dims->data[0], 1);

  int D_in = tokens.dims->data[2];
  int D_mid = gate_weights.dims->data[0];
  int E = gate_weights.dims->data[1];

  TF_LITE_ENSURE(context, tflite::NumElements(&scale) == 1 ||
                              tflite::NumElements(&scale) == E);

  TF_LITE_ENSURE_EQ(context, gate_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, gate_weights.dims->data[3], D_in);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[0], D_mid);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[1], E);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[3], D_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[0], D_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[1], E);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[3], D_mid);

  return kTfLiteOk;
}

TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           uint32_t& next_external_id, const NodeInfo& node,
                           std::vector<std::unique_ptr<float[]>>* temp_buffers,
                           MoeInfo* moe_info, bool static_shape) {
  if (moe_info != nullptr && !node.inputs.empty()) {
    moe_info->tokens_tensor_index = node.inputs[0];
  }

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

  TF_LITE_ENSURE_EQ(context, tokens.dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, routing_weights.dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, expert_indices.dims->data[0], 1);

  int D_in = tokens.dims->data[2];
  int D_out = D_in;
  int K = expert_indices.dims->data[2];

  int D_mid = gate_weights.dims->data[0];
  int E = gate_weights.dims->data[1];

  TF_LITE_ENSURE(context, tflite::NumElements(&scale) == 1 ||
                              tflite::NumElements(&scale) == E);

  if (moe_info != nullptr) {
    moe_info->expert_indices_tensor_index = node.inputs[2];
    moe_info->num_experts = E;
    moe_info->k = K;
    moe_info->dynamic_routing_bufs.resize(E + 1);
  }

  TF_LITE_ENSURE_EQ(context, gate_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, gate_weights.dims->data[3], D_in);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[0], D_mid);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[1], E);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[3], D_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[0], D_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[1], E);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[2], 1);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[3], D_mid);

  // Transpose weights.
  auto W_gate_buf = std::make_unique<float[]>(E * D_in * D_mid);
  TransposeWeightsGateUp(reinterpret_cast<const float*>(gate_weights.data.raw),
                         W_gate_buf.get(), E, D_mid, D_in);

  auto W_up_buf = std::make_unique<float[]>(E * D_in * D_mid);
  TransposeWeightsGateUp(reinterpret_cast<const float*>(up_weights.data.raw),
                         W_up_buf.get(), E, D_mid, D_in);

  auto W_down_buf = std::make_unique<float[]>(E * D_mid * D_in);
  size_t num_scale_elements = tflite::NumElements(&scale);
  TransposeAndScaleWeightsDown(
      reinterpret_cast<const float*>(down_weights.data.raw),
      reinterpret_cast<const float*>(scale.data.raw), num_scale_elements,
      W_down_buf.get(), E, D_in, D_mid);

  float* w_gate_ptr = W_gate_buf.get();
  float* w_up_ptr = W_up_buf.get();
  float* w_down_ptr = W_down_buf.get();

  temp_buffers->push_back(std::move(W_gate_buf));
  temp_buffers->push_back(std::move(W_up_buf));
  temp_buffers->push_back(std::move(W_down_buf));

  uint32_t tokens_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[0]);
  uint32_t routing_weights_val =
      GetOrCreateValueId(context, subgraph, tensor_to_value_id, node.inputs[1]);
  uint32_t output_val = YNN_INVALID_VALUE_ID;
  auto out_it = tensor_to_value_id.find(node.outputs[0]);
  if (out_it != tensor_to_value_id.end()) {
    output_val = out_it->second;
  }

  uint32_t tokens_reshaped = YNN_INVALID_VALUE_ID;
  size_t shape_N_Din[] = {0, static_cast<size_t>(D_in)};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 2, shape_N_Din, tokens_val, &tokens_reshaped, 0));

  uint32_t routing_weights_reshaped = YNN_INVALID_VALUE_ID;
  size_t shape_N_K[] = {0, static_cast<size_t>(K)};
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_static_reshape(subgraph, 2, shape_N_K, routing_weights_val,
                                &routing_weights_reshaped, 0));

  std::vector<uint32_t> expert_out_ids(E);
  for (int e = 0; e < E; ++e) {
    uint32_t w_gate_e = YNN_INVALID_VALUE_ID;
    uint32_t w_up_e = YNN_INVALID_VALUE_ID;
    uint32_t w_down_e = YNN_INVALID_VALUE_ID;

    size_t w_gate_dims[] = {static_cast<size_t>(D_in),
                            static_cast<size_t>(D_mid)};
    size_t w_down_dims[] = {static_cast<size_t>(D_mid),
                            static_cast<size_t>(D_out)};

    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_fp32, 2, w_gate_dims,
                          w_gate_ptr + e * D_in * D_mid, 0, &w_gate_e));
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_fp32, 2, w_gate_dims,
                          w_up_ptr + e * D_in * D_mid, 0, &w_up_e));
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_fp32, 2, w_down_dims,
                          w_down_ptr + e * D_mid * D_out, 0, &w_down_e));

    uint32_t exp_token_id = next_external_id++;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_int32, 1, nullptr, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &exp_token_id));
    if (moe_info != nullptr) {
      moe_info->dynamic_routing_val_ids.push_back(exp_token_id);
    }

    uint32_t tokens_e = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(DefineGather2D(subgraph, tokens_reshaped,
                                             exp_token_id, D_in, &tokens_e));

    uint32_t gate_e = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_e, w_gate_e,
                                             YNN_INVALID_VALUE_ID, &gate_e, 0));

    uint32_t gelu_gate = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn::define_approx_gelu(subgraph, gate_e, gelu_gate));

    uint32_t up_e = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, tokens_e, w_up_e,
                                             YNN_INVALID_VALUE_ID, &up_e, 0));

    uint32_t mid_e = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                gelu_gate, up_e, &mid_e, 0));

    uint32_t out_e = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, 1, mid_e, w_down_e,
                                             YNN_INVALID_VALUE_ID, &out_e, 0));

    expert_out_ids[e] = out_e;
  }

  uint32_t all_experts_out = YNN_INVALID_VALUE_ID;
  if (E == 1) {
    all_experts_out = expert_out_ids[0];
  } else {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_concatenate(
        subgraph, 0, E, expert_out_ids.data(), &all_experts_out, 0));
  }

  uint32_t reverse_indices_id = next_external_id++;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_tensor(subgraph, ynn_type_int32, 1, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT, &reverse_indices_id));
  if (moe_info != nullptr) {
    moe_info->dynamic_routing_val_ids.push_back(reverse_indices_id);
  }

  uint32_t dispatched_out_2d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(DefineGather2D(subgraph, all_experts_out,
                                           reverse_indices_id, D_out,
                                           &dispatched_out_2d));

  uint32_t dispatched_out_3d = YNN_INVALID_VALUE_ID;
  size_t shape_N_K_Dout[] = {0, static_cast<size_t>(K),
                             static_cast<size_t>(D_out)};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 3, shape_N_K_Dout, dispatched_out_2d, &dispatched_out_3d, 0));

  uint32_t rw_expanded = YNN_INVALID_VALUE_ID;
  int32_t expand_axis_2 = 2;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 1, &expand_axis_2, routing_weights_reshaped, &rw_expanded, 0));

  uint32_t weighted_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                              dispatched_out_3d, rw_expanded,
                                              &weighted_out, 0));

  uint32_t reduced_out = YNN_INVALID_VALUE_ID;
  int32_t axis_1 = 1;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &axis_1, weighted_out,
                        YNN_INVALID_VALUE_ID, &reduced_out, 0));

  size_t shape_1_N_Dout[] = {1, 0, static_cast<size_t>(D_out)};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, 3, shape_1_N_Dout, reduced_out, &output_val, 0));

  tensor_to_value_id[node.outputs[0]] = output_val;

  return kTfLiteOk;
}

TfLiteStatus InitMoeRuntime(TfLiteContext* context, ynn_runtime_t runtime,
                            const std::vector<MoeInfo>& moe_infos) {
  for (const auto& moe_info : moe_infos) {
    const TfLiteTensor& expert_indices =
        context->tensors[moe_info.expert_indices_tensor_index];
    size_t N = (expert_indices.dims && expert_indices.dims->size >= 2)
                   ? expert_indices.dims->data[1]
                   : 1;
    size_t NK = N * moe_info.k;
    size_t init_count = std::max<size_t>(1, NK / moe_info.num_experts);
    for (int e = 0; e < moe_info.num_experts; ++e) {
      size_t init_dim[] = {init_count};
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
          runtime, moe_info.dynamic_routing_val_ids[e], 1, init_dim));
    }
    size_t nk_dims[] = {NK};
    TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
        runtime, moe_info.dynamic_routing_val_ids[moe_info.num_experts], 1,
        nk_dims));
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMoeNodes(TfLiteContext* context, ynn_runtime_t runtime,
                          std::vector<MoeInfo>& moe_infos) {
  for (auto& moe_info : moe_infos) {
    const TfLiteTensor& tokens_tensor =
        context->tensors[moe_info.tokens_tensor_index];
    const TfLiteTensor& expert_indices =
        context->tensors[moe_info.expert_indices_tensor_index];
    const int32_t* ei_data =
        reinterpret_cast<const int32_t*>(expert_indices.data.raw);
    size_t N = (tokens_tensor.dims && tokens_tensor.dims->size >= 2)
                   ? tokens_tensor.dims->data[tokens_tensor.dims->size - 2]
                   : 1;
    size_t K = moe_info.k;
    size_t E = moe_info.num_experts;
    size_t NK = N * K;

    std::vector<std::vector<int32_t>> expert_to_token_idx(E);
    for (size_t i = 0; i < N; ++i) {
      for (size_t k = 0; k < K; ++k) {
        int32_t exp = ei_data[i * K + k];
        if (exp >= 0 && static_cast<size_t>(exp) < E) {
          expert_to_token_idx[exp].push_back(static_cast<int32_t>(i));
        }
      }
    }

    std::vector<size_t> expert_slice_sizes(E);
    for (size_t e = 0; e < E; ++e) {
      size_t count = std::max<size_t>(1, expert_to_token_idx[e].size());
      expert_slice_sizes[e] = count;
      moe_info.dynamic_routing_bufs[e].resize(count);
      if (expert_to_token_idx[e].empty()) {
        moe_info.dynamic_routing_bufs[e][0] = 0;
      } else {
        for (size_t c = 0; c < expert_to_token_idx[e].size(); ++c) {
          moe_info.dynamic_routing_bufs[e][c] = expert_to_token_idx[e][c];
        }
      }
      size_t count_dim[] = {count};
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
          runtime, moe_info.dynamic_routing_val_ids[e], 1, count_dim));
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
          runtime, moe_info.dynamic_routing_val_ids[e],
          moe_info.dynamic_routing_bufs[e].data()));
    }

    std::vector<size_t> expert_cursor(E, 0);
    std::vector<size_t> expert_base(E, 0);
    for (size_t e = 1; e < E; ++e) {
      expert_base[e] = expert_base[e - 1] + expert_slice_sizes[e - 1];
    }
    moe_info.dynamic_routing_bufs[E].resize(NK);
    for (size_t i = 0; i < N; ++i) {
      for (size_t k = 0; k < K; ++k) {
        int32_t exp = ei_data[i * K + k];
        size_t pos = 0;
        if (exp >= 0 && static_cast<size_t>(exp) < E) {
          pos = expert_base[exp] + expert_cursor[exp]++;
        }
        moe_info.dynamic_routing_bufs[E][i * K + k] = static_cast<int32_t>(pos);
      }
    }
    size_t nk_dims[] = {NK};
    TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
        runtime, moe_info.dynamic_routing_val_ids[E], 1, nk_dims));
    TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
        runtime, moe_info.dynamic_routing_val_ids[E],
        moe_info.dynamic_routing_bufs[E].data()));
  }
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
