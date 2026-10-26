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
#include <limits>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {
namespace {

// The MoE op comes in two input layouts:
//  - the base layout: tokens, routing weights, expert indices, gate/up/down
//    weights and the output scale.
//  - the extended layout, which additionally carries an explicit per-expert
//    scale tensor for each of the gate/up/down weights.
constexpr int kNumInputsWithoutWeightScales = 7;
constexpr int kNumInputsWithWeightScales = 10;

// Positions of the MoE inputs in `node->inputs`. The `*_scale` indices are
// `-1` unless the op uses the extended layout.
struct MoeInputIndices {
  int tokens;
  int routing_weights;
  int expert_indices;
  int gate;
  int gate_scale;
  int up;
  int up_scale;
  int down;
  int down_scale;
  int scale;
};

MoeInputIndices GetInputIndices(bool has_weight_scales) {
  return MoeInputIndices{
      /*tokens=*/0,
      /*routing_weights=*/1,
      /*expert_indices=*/2,
      /*gate=*/3,
      /*gate_scale=*/has_weight_scales ? 4 : -1,
      /*up=*/has_weight_scales ? 5 : 4,
      /*up_scale=*/has_weight_scales ? 6 : -1,
      /*down=*/has_weight_scales ? 7 : 5,
      /*down_scale=*/has_weight_scales ? 8 : -1,
      /*scale=*/has_weight_scales ? 9 : 6,
  };
}

bool Is16BitFloatType(TfLiteType type) {
  return type == kTfLiteFloat16 || type == kTfLiteBFloat16;
}

bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || Is16BitFloatType(type);
}

// Weight types that make the MoE subgraph run in dynamically quantized mode.
bool IsQuantizedWeightType(TfLiteType type) {
  return type == kTfLiteInt8 || type == kTfLiteInt4 || type == kTfLiteInt2;
}

// Converts `value_id` to `to`, or leaves it alone if it already has that type.
TfLiteStatus ConvertIfNeeded(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TfLiteType from, TfLiteType to,
                             uint32_t& value_id) {
  if (from == to) {
    return kTfLiteOk;
  }
  uint32_t converted = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_convert(subgraph, value_id, GetYnnType(to), &converted, 0));
  value_id = converted;
  return kTfLiteOk;
}

// One of the gate/up/down weight tensors, together with the graph values
// derived from it.
struct WeightValues {
  int tensor_index;
  // Index of the explicit per-expert scale tensor, or `-1` if the op does not
  // carry one.
  int scale_tensor_index;
  // Weights, after transposition to [E, D_in, D_out].
  uint32_t value_id = YNN_INVALID_VALUE_ID;
  // Quantization parameters, invalid for float weights.
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;

  bool is_blockwise = false;
  size_t block_size = 0;
};

// The three weight tensors, in the order they appear in `node->inputs` and in
// the order the subgraph consumes them.
enum WeightKind { kGate = 0, kUp = 1, kDown = 2 };
constexpr int kNumWeights = 3;

// Whether the weight tensor `w` is supported. `scale_tensor_index` is the
// explicit scale tensor of `w`, or -1 if the op does not carry one. With
// explicit scales, the lowering ignores the quantization of `w`, which then
// may also be a multi-axis quantization that restates the explicit scales, per
// output channel and expert.
bool IsWeightTensorSupported(TfLiteContext* context, const TfLiteTensor& w,
                             int scale_tensor_index,
                             bool allow_blockwise = false) {
  if (w.quantization.type == kTfLiteBlockwiseQuantization) {
    return allow_blockwise && IsTensorSupported(w, /*allow_per_channel=*/true,
                                                /*allow_blockwise=*/true);
  }
  if (w.quantization.type != kTfLiteMultiAxisQuantization) {
    return IsTensorSupported(w, /*allow_per_channel=*/true,
                             /*allow_blockwise=*/allow_blockwise);
  }
  const auto* params =
      static_cast<const TfLiteMultiAxisQuantization*>(w.quantization.params);
  if (scale_tensor_index < 0 || params == nullptr ||
      params->scales != scale_tensor_index || params->blocksize != 0 ||
      params->quantized_dimensions == nullptr ||
      params->quantized_dimensions->size != 2 ||
      params->quantized_dimensions->data[0] != 0 ||
      params->quantized_dimensions->data[1] != 1) {
    return false;
  }
  int resolved_zp_idx = -1;
  if (ResolveZeroPointTensorIndex(context, w, params->scales,
                                  params->zero_points,
                                  &resolved_zp_idx) != kTfLiteOk ||
      resolved_zp_idx != -1) {
    return false;
  }
  TfLiteTensor unquantized = w;
  unquantized.quantization = {kTfLiteNoQuantization, nullptr};
  unquantized.params = {};
  return IsTensorSupported(unquantized);
}

}  // namespace

bool IsMoe(const TfLiteRegistration* registration, const TfLiteNode* node) {
  if (registration == nullptr || node == nullptr ||
      registration->builtin_code != kTfLiteBuiltinStablehloComposite ||
      node->builtin_data == nullptr) {
    return false;
  }
  const auto* composite_params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  return composite_params->name != nullptr &&
         strcmp(composite_params->name, "odml.moe_experts") == 0;
}

TfLiteStatus IsMoeSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context,
                            const TfLiteYNNPackDelegateOptions& options) {
  TF_LITE_ENSURE(context, IsMoe(registration, node));
  TF_LITE_ENSURE(context, node->inputs != nullptr && node->outputs != nullptr);
  TF_LITE_ENSURE(context, node->inputs->size == kNumInputsWithoutWeightScales ||
                              node->inputs->size == kNumInputsWithWeightScales);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  for (int i = 0; i < node->inputs->size; ++i) {
    const int idx = node->inputs->data[i];
    TF_LITE_ENSURE(context, idx >= 0 && idx < context->tensors_size);
  }
  const int out_idx = node->outputs->data[0];
  TF_LITE_ENSURE(context, out_idx >= 0 && out_idx < context->tensors_size);

  const bool has_weight_scales =
      node->inputs->size == kNumInputsWithWeightScales;
  const MoeInputIndices idx = GetInputIndices(has_weight_scales);

  const TfLiteTensor& tokens = context->tensors[node->inputs->data[idx.tokens]];
  const TfLiteTensor& routing_weights =
      context->tensors[node->inputs->data[idx.routing_weights]];
  const TfLiteTensor& expert_indices =
      context->tensors[node->inputs->data[idx.expert_indices]];
  const TfLiteTensor& gate_weights =
      context->tensors[node->inputs->data[idx.gate]];
  const TfLiteTensor& up_weights = context->tensors[node->inputs->data[idx.up]];
  const TfLiteTensor& down_weights =
      context->tensors[node->inputs->data[idx.down]];
  const TfLiteTensor& scale = context->tensors[node->inputs->data[idx.scale]];
  const TfLiteTensor& output = context->tensors[out_idx];

  TF_LITE_ENSURE(context, IsTensorSupported(tokens));
  TF_LITE_ENSURE(context, IsTensorSupported(routing_weights));
  TF_LITE_ENSURE(context, IsTensorSupported(expert_indices));
  TF_LITE_ENSURE(context, IsTensorSupported(scale));
  TF_LITE_ENSURE(context, IsTensorSupported(output));

  TF_LITE_ENSURE(context, tflite::IsConstantTensor(&scale));

  TF_LITE_ENSURE(context, IsFloatType(tokens.type));
  TF_LITE_ENSURE(context, IsFloatType(routing_weights.type));
  TF_LITE_ENSURE_EQ(context, expert_indices.type, kTfLiteInt32);
  TF_LITE_ENSURE(context, IsFloatType(scale.type));
  TF_LITE_ENSURE(context, IsFloatType(output.type));
  TF_LITE_ENSURE_EQ(context, tokens.type, output.type);

  const TfLiteTensor* weights[] = {&gate_weights, &up_weights, &down_weights};
  const int weight_scale_indices[] = {idx.gate_scale, idx.up_scale,
                                      idx.down_scale};
  for (int i = 0; i < kNumWeights; ++i) {
    const TfLiteTensor* w = weights[i];
    const int scale_tensor_index =
        has_weight_scales ? node->inputs->data[weight_scale_indices[i]] : -1;
    TF_LITE_ENSURE(context, IsWeightTensorSupported(
                                context, *w, scale_tensor_index,
                                options.enable_blockwise_quantization));
    TF_LITE_ENSURE(context, tflite::IsConstantTensor(w));
    TF_LITE_ENSURE(context,
                   IsFloatType(w->type) || IsQuantizedWeightType(w->type));
    TF_LITE_ENSURE_EQ(context, w->type, gate_weights.type);
    TF_LITE_ENSURE(context, w->dims != nullptr && w->dims->size == 4);
    TF_LITE_ENSURE_EQ(context, w->dims->data[2], 1);

    if (w->quantization.type == kTfLiteBlockwiseQuantization) {
      TF_LITE_ENSURE(context, !has_weight_scales);
      TF_LITE_ENSURE(context, options.enable_blockwise_quantization);
      const auto* bq_params = static_cast<const TfLiteBlockwiseQuantization*>(
          w->quantization.params);
      TF_LITE_ENSURE(context, bq_params != nullptr);
      TF_LITE_ENSURE(context, bq_params->blocksize > 0);
      TF_LITE_ENSURE(context, w->dims->data[3] % bq_params->blocksize == 0);

      int resolved_zp_idx = -1;
      TF_LITE_ENSURE_STATUS(
          ResolveZeroPointTensorIndex(context, *w, bq_params->scale,
                                      bq_params->zero_point, &resolved_zp_idx));
      const TfLiteTensor& scale_tensor = context->tensors[bq_params->scale];
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->size, 4);
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->data[0], w->dims->data[0]);
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->data[1], w->dims->data[1]);
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->data[2], 1);
      TF_LITE_ENSURE_EQ(context, scale_tensor.dims->data[3],
                        w->dims->data[3] / bq_params->blocksize);
    } else if (!has_weight_scales &&
               w->quantization.type == kTfLiteAffineQuantization) {
      const auto* aq_params =
          static_cast<const TfLiteAffineQuantization*>(w->quantization.params);
      TF_LITE_ENSURE(context, aq_params != nullptr &&
                                  aq_params->scale != nullptr &&
                                  aq_params->scale->size >= 1);
      if (aq_params->scale->size > 1) {
        TF_LITE_ENSURE_EQ(context, aq_params->quantized_dimension, 0);
        TF_LITE_ENSURE_EQ(context, aq_params->scale->size, w->dims->data[0]);
        TF_LITE_ENSURE(context, aq_params->zero_point != nullptr &&
                                    (aq_params->zero_point->size == 1 ||
                                     aq_params->zero_point->size ==
                                         aq_params->scale->size));
      }
      if (aq_params->zero_point != nullptr) {
        for (int z = 0; z < aq_params->zero_point->size; ++z) {
          TF_LITE_ENSURE_EQ(context, aq_params->zero_point->data[z], 0);
        }
      }
    } else if (!has_weight_scales && IsQuantizedWeightType(w->type)) {
      TF_LITE_ENSURE(context,
                     w->params.scale != 0.0f && w->params.zero_point == 0);
    }
  }

  const int num_experts = gate_weights.dims->data[1];
  const int d_mid = gate_weights.dims->data[0];
  const int d_in = gate_weights.dims->data[3];
  TF_LITE_ENSURE(context, num_experts > 0 && d_mid > 0 && d_in > 0);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[0], d_mid);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[1], num_experts);
  TF_LITE_ENSURE_EQ(context, up_weights.dims->data[3], d_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[0], d_in);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[1], num_experts);
  TF_LITE_ENSURE_EQ(context, down_weights.dims->data[3], d_mid);

  const size_t num_scale_elements = tflite::NumElements(&scale);
  TF_LITE_ENSURE(context,
                 num_scale_elements == 1 ||
                     num_scale_elements == static_cast<size_t>(num_experts));

  if (has_weight_scales) {
    TF_LITE_ENSURE(context, IsQuantizedWeightType(gate_weights.type));
    const int scale_indices[] = {idx.gate_scale, idx.up_scale, idx.down_scale};
    for (int i = 0; i < kNumWeights; ++i) {
      const TfLiteTensor& weight_scale =
          context->tensors[node->inputs->data[scale_indices[i]]];
      TF_LITE_ENSURE(context, IsTensorSupported(weight_scale));
      TF_LITE_ENSURE(context, tflite::IsConstantTensor(&weight_scale));
      TF_LITE_ENSURE(context, IsFloatType(weight_scale.type));
      TF_LITE_ENSURE(context, weight_scale.dims != nullptr &&
                                  weight_scale.dims->size == 4);
      TF_LITE_ENSURE_EQ(context, weight_scale.dims->data[0],
                        weights[i]->dims->data[0]);
      TF_LITE_ENSURE_EQ(context, weight_scale.dims->data[1], num_experts);
      TF_LITE_ENSURE_EQ(context, weight_scale.dims->data[2], 1);
      TF_LITE_ENSURE_EQ(context, weight_scale.dims->data[3], 1);
    }
  }

  TF_LITE_ENSURE(context, tokens.dims != nullptr && tokens.dims->size == 3);
  TF_LITE_ENSURE(context, routing_weights.dims != nullptr &&
                              routing_weights.dims->size == 3);
  TF_LITE_ENSURE(context, expert_indices.dims != nullptr &&
                              expert_indices.dims->size == 3);
  TF_LITE_ENSURE(context, output.dims != nullptr && output.dims->size == 3);
  TF_LITE_ENSURE_EQ(context, tokens.dims->data[2], d_in);
  TF_LITE_ENSURE_EQ(context, output.dims->data[2], d_in);
  TF_LITE_ENSURE(context, tokens.dims->data[0] > 0 && tokens.dims->data[1] > 0);
  TF_LITE_ENSURE(context, expert_indices.dims->data[2] > 0);
  TF_LITE_ENSURE_EQ(context, routing_weights.dims->data[2],
                    expert_indices.dims->data[2]);
  TF_LITE_ENSURE_EQ(context, routing_weights.dims->data[0],
                    tokens.dims->data[0]);
  TF_LITE_ENSURE_EQ(context, routing_weights.dims->data[1],
                    tokens.dims->data[1]);
  TF_LITE_ENSURE_EQ(context, expert_indices.dims->data[0],
                    tokens.dims->data[0]);
  TF_LITE_ENSURE_EQ(context, expert_indices.dims->data[1],
                    tokens.dims->data[1]);
  TF_LITE_ENSURE_EQ(context, output.dims->data[0], tokens.dims->data[0]);
  TF_LITE_ENSURE_EQ(context, output.dims->data[1], tokens.dims->data[1]);

  // Sub-byte weights pack several values into one byte along the innermost
  // dimension. `IsTensorSupported` checks that for the layout the weights
  // arrive in, but `DefineMoeNode` transposes [X, E, 1, Y] to [E, Y, X], which
  // makes the leading dimension the innermost one, so it has to hold a whole
  // number of bytes too.
  const int weights_per_byte =
      static_cast<int>(YnnTypeElementCount(GetYnnType(gate_weights.type)));
  if (weights_per_byte > 1) {
    for (const TfLiteTensor* w : weights) {
      TF_LITE_ENSURE_EQ(context, w->dims->data[0] % weights_per_byte, 0);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           uint32_t& next_external_id, MoeDispatch& dispatch,
                           const NodeInfo& node) {
  TF_LITE_ENSURE(context, node.inputs.size() == kNumInputsWithoutWeightScales ||
                              node.inputs.size() == kNumInputsWithWeightScales);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  const bool has_weight_scales =
      node.inputs.size() == kNumInputsWithWeightScales;
  const MoeInputIndices idx = GetInputIndices(has_weight_scales);

  const TfLiteTensor& tokens = context->tensors[node.inputs[idx.tokens]];
  const TfLiteTensor& routing_weights =
      context->tensors[node.inputs[idx.routing_weights]];
  const TfLiteTensor& expert_indices =
      context->tensors[node.inputs[idx.expert_indices]];
  const TfLiteTensor& gate_weights = context->tensors[node.inputs[idx.gate]];
  const TfLiteTensor& scale = context->tensors[node.inputs[idx.scale]];

  TF_LITE_ENSURE(context, IsFloatType(tokens.type));
  TF_LITE_ENSURE(context, IsFloatType(routing_weights.type));
  TF_LITE_ENSURE_EQ(context, expert_indices.type, kTfLiteInt32);
  TF_LITE_ENSURE(context, IsFloatType(scale.type));

  const int num_experts = gate_weights.dims->data[1];
  TF_LITE_ENSURE(context, dispatch.batch_size > 0 && dispatch.top_k > 0);

  TF_LITE_ENSURE(context, tflite::NumElements(&scale) == 1 ||
                              tflite::NumElements(&scale) == num_experts);

  const bool is_quantized = IsQuantizedWeightType(gate_weights.type);

  // A float dot needs both operands in the same type: YNNPACK has fp32, fp16
  // and bf16 kernels, all of which accumulate in fp32, but none for a mixed
  // pair.
  // TODO: If YNNPACK adds mixed-type float dot kernels, remove this conversion.
  const TfLiteType dot_float_type =
      tokens.type == gate_weights.type ? tokens.type : kTfLiteFloat32;

  WeightValues weights[kNumWeights] = {
      {node.inputs[idx.gate],
       has_weight_scales ? node.inputs[idx.gate_scale] : -1},
      {node.inputs[idx.up], has_weight_scales ? node.inputs[idx.up_scale] : -1},
      {node.inputs[idx.down],
       has_weight_scales ? node.inputs[idx.down_scale] : -1},
  };

  uint32_t scale_val = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          node.inputs[idx.scale]);

  // 1. Transpose constant weights from 4D [*, E, 1, *] to 3D [E, *, *],
  // dropping the unit axis 2 so the expert dimension E is on axis 0 and the
  // reduction dimension is on the second-to-last axis.
  const int32_t perm_3d[] = {1, 3, 0};
  constexpr size_t kRankWeights = 3;
  for (WeightValues& w : weights) {
    const uint32_t weights_val = GetOrCreateValueId(
        context, subgraph, tensor_to_value_id, w.tensor_index);
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, kRankWeights, perm_3d, weights_val, &w.value_id, 0));
  }

  if (is_quantized) {
    for (WeightValues& w : weights) {
      const TfLiteTensor& tensor = context->tensors[w.tensor_index];
      if (tensor.quantization.type == kTfLiteBlockwiseQuantization) {
        w.is_blockwise = true;
        const auto* bq_params = static_cast<const TfLiteBlockwiseQuantization*>(
            tensor.quantization.params);
        w.block_size = bq_params->blocksize;
        w.scale_tensor_index = bq_params->scale;
        int resolved_zp_idx = -1;
        TF_LITE_ENSURE_STATUS(ResolveZeroPointTensorIndex(
            context, tensor, bq_params->scale, bq_params->zero_point,
            &resolved_zp_idx));
        // Drop the unit axis 2 from 4D [N, E, 1, K / block_size].
        const int32_t perm_scale[] = {1, 0, 3};
        if (resolved_zp_idx >= 0) {
          const uint32_t raw_zp_val = GetOrCreateValueId(
              context, subgraph, tensor_to_value_id, resolved_zp_idx);
          TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
              subgraph, 3, perm_scale, raw_zp_val, &w.zero_point_id, 0));
        }

        const uint32_t raw_scale_val = GetOrCreateValueId(
            context, subgraph, tensor_to_value_id, w.scale_tensor_index);
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
            subgraph, 3, perm_scale, raw_scale_val, &w.scale_id, 0));
      } else if (has_weight_scales) {
        const uint32_t weight_scale_val = GetOrCreateValueId(
            context, subgraph, tensor_to_value_id, w.scale_tensor_index);
        const int32_t perm_2d[] = {1, 0};
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
            subgraph, 2, perm_2d, weight_scale_val, &w.scale_id, 0));
      } else {
        TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
            context, subgraph, tensor, &w.scale_id, &w.zero_point_id));
        TF_LITE_ENSURE_EQ(context, w.zero_point_id, YNN_INVALID_VALUE_ID);
      }
    }
  }

  uint32_t tokens_val = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, node.inputs[idx.tokens]);
  uint32_t routing_weights_val = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, node.inputs[idx.routing_weights]);
  uint32_t expert_indices_val = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, node.inputs[idx.expert_indices]);
  uint32_t output_val = YNN_INVALID_VALUE_ID;
  auto out_it = tensor_to_value_id.find(node.outputs[0]);
  if (out_it != tensor_to_value_id.end()) {
    output_val = out_it->second;
  }

  // 2. Routing weights & scale: [B, N, K]
  uint32_t scale_1d = YNN_INVALID_VALUE_ID;
  if (scale.dims != nullptr && scale.dims->size > 1) {
    int32_t perm_last_axis[] = {scale.dims->size - 1};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 1, perm_last_axis, scale_val, &scale_1d, 0));
  } else {
    scale_1d = scale_val;
  }

  int32_t gather_axis_0 = 0;
  uint32_t rw_scaled = routing_weights_val;
  size_t num_scale_elements = tflite::NumElements(&scale);
  if (num_scale_elements == 1) {
    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                routing_weights_val, scale_1d,
                                                &rw_scaled, 0));
  } else if (num_scale_elements == static_cast<size_t>(num_experts)) {
    uint32_t scale_gathered = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 3,
                                                scale_1d, expert_indices_val,
                                                &scale_gathered, 0));

    rw_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                routing_weights_val,
                                                scale_gathered, &rw_scaled, 0));
  }

  if (!is_quantized) {
    for (WeightValues& w : weights) {
      TF_LITE_ENSURE_STATUS(ConvertIfNeeded(
          context, subgraph, context->tensors[w.tensor_index].type,
          dot_float_type, w.value_id));
    }
  }

  auto dynamic_quantize = [&](uint32_t input_id, uint32_t* quant_id,
                              uint32_t* zp_id,
                              uint32_t* scale_id) -> TfLiteStatus {
    int32_t reduce_axis_last = -1;
    uint32_t min_max_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
        subgraph, ynn_reduce_min_max, 1, &reduce_axis_last, input_id,
        YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dynamic_quantization(
        subgraph, min_max_id, ynn_type_int8, zp_id, scale_id, 0));

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(
        subgraph, input_id, ynn_type_int8, *zp_id, *scale_id, quant_id, 0));
    return kTfLiteOk;
  };

  // 3. Define external inputs for host-partitioned `(token, expert)` routing
  // so each expert `e` computes a contiguous run of rows via
  // `ynn_define_ragged_dot`.
  dispatch.expert_indices_tensor_index = node.inputs[idx.expert_indices];
  dispatch.num_experts = num_experts;
  dispatch.slot_token_id = next_external_id++;
  dispatch.slot_expert_id = next_external_id++;
  dispatch.token_slot_id = next_external_id++;

  const size_t slot_dims[2] = {0, 1};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
      subgraph, ynn_type_int32, 2, slot_dims, nullptr,
      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dispatch.slot_token_id));
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
      subgraph, ynn_type_int32, 2, slot_dims, nullptr,
      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dispatch.slot_expert_id));
  const size_t token_slot_dims[3] = {0, 0, 1};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
      subgraph, ynn_type_int32, 3, token_slot_dims, nullptr,
      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dispatch.token_slot_id));

  // [B, N, D_in] -> [T, D_in]
  uint32_t tokens_flat = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_fuse_dim(subgraph, 0, 2, tokens_val, &tokens_flat, 0));

  // Quantize the tokens before gathering them, so that each token is only
  // quantized once. [T, D_in] gathered by [R, 1] -> [R, D_in], and the
  // quantization parameters [T, 1] -> [R, 1].
  auto gather_slots = [&](uint32_t input, uint32_t* output) -> TfLiteStatus {
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 2,
                                                input, dispatch.slot_token_id,
                                                output, 0));
    return kTfLiteOk;
  };
  uint32_t a = YNN_INVALID_VALUE_ID;
  uint32_t a_zp = YNN_INVALID_VALUE_ID;
  uint32_t a_scale = YNN_INVALID_VALUE_ID;
  if (is_quantized) {
    uint32_t tokens_q = YNN_INVALID_VALUE_ID;
    uint32_t tokens_zp = YNN_INVALID_VALUE_ID;
    uint32_t tokens_scale = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(
        dynamic_quantize(tokens_flat, &tokens_q, &tokens_zp, &tokens_scale));
    TF_LITE_ENSURE_STATUS(gather_slots(tokens_q, &a));
    TF_LITE_ENSURE_STATUS(gather_slots(tokens_zp, &a_zp));
    TF_LITE_ENSURE_STATUS(gather_slots(tokens_scale, &a_scale));
  } else {
    TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, tokens.type,
                                          dot_float_type, tokens_flat));
    TF_LITE_ENSURE_STATUS(gather_slots(tokens_flat, &a));
  }

  auto define_ragged_dot = [&](uint32_t a_id, uint32_t a_zp_id,
                               uint32_t a_scale_id, const WeightValues& w,
                               uint32_t* out_id) -> TfLiteStatus {
    if (!is_quantized) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_ragged_dot(
          subgraph, /*num_k_dims=*/1, a_id, w.value_id, YNN_INVALID_VALUE_ID,
          dispatch.slot_expert_id, out_id, 0));
      return kTfLiteOk;
    }

    const uint32_t dot_flags =
        (context->tensors[w.tensor_index].type == kTfLiteInt8 &&
         w.zero_point_id == YNN_INVALID_VALUE_ID)
            ? YNN_NODE_FLAG_SYMMETRIC_B
            : 0;
    if (w.is_blockwise) {
      TF_LITE_ENSURE_YNN_STATUS(ynn::define_blockwise_dot(
          subgraph, a_id, a_zp_id, a_scale_id, w.value_id, w.zero_point_id,
          w.scale_id, w.block_size, /*bias_id=*/YNN_INVALID_VALUE_ID,
          ynn_type_fp32, *out_id, dot_flags, dispatch.slot_expert_id));
      return kTfLiteOk;
    }

    // Compute per-expert column sums sum_b [E, N] = sum_k b[e, k, n].
    int32_t reduce_axis_k = -2;
    uint32_t sum_b = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &reduce_axis_k,
                          w.value_id, YNN_INVALID_VALUE_ID, &sum_b, 0));

    uint32_t sum_b_gathered = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 2,
                                                sum_b, dispatch.slot_expert_id,
                                                &sum_b_gathered, 0));

    uint32_t dot_zp = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_multiply, a_zp_id, sum_b_gathered, &dot_zp, 0));
    const int32_t zero = 0;
    uint32_t zero_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_int32, 0, nullptr, &zero,
                          YNN_VALUE_FLAG_COPY_DATA, &zero_id));
    uint32_t accum_init = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_subtract, zero_id, dot_zp, &accum_init, 0));

    uint32_t dot_out = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_ragged_dot(
        subgraph, /*num_k_dims=*/1, a_id, w.value_id, accum_init,
        dispatch.slot_expert_id, &dot_out, dot_flags));

    uint32_t dot_fp32 = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_convert(subgraph, dot_out, ynn_type_fp32, &dot_fp32, 0));

    uint32_t dot_scaled = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_multiply, dot_fp32, a_scale_id, &dot_scaled, 0));

    uint32_t w_scale_for_mul = w.scale_id;
    if (has_weight_scales) {
      w_scale_for_mul = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_gather(subgraph, 1, &gather_axis_0, 2, w.scale_id,
                            dispatch.slot_expert_id, &w_scale_for_mul, 0));
    }

    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_multiply, dot_scaled, w_scale_for_mul, out_id, 0));
    return kTfLiteOk;
  };

  // 4. [R, D_in] x [E, D_in, *] -> [R, *] via ragged_dot.
  uint32_t gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(
      define_ragged_dot(a, a_zp, a_scale, weights[kGate], &gate));

  uint32_t act_gate = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn::define_approx_gelu(subgraph, gate, act_gate));

  uint32_t up = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(define_ragged_dot(a, a_zp, a_scale, weights[kUp], &up));

  uint32_t mid = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_multiply, act_gate, up, &mid, 0));

  uint32_t mid_in = mid;
  uint32_t mid_zp = YNN_INVALID_VALUE_ID;
  uint32_t mid_scale = YNN_INVALID_VALUE_ID;
  if (is_quantized) {
    mid_in = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(dynamic_quantize(mid, &mid_in, &mid_zp, &mid_scale));
  } else {
    TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, kTfLiteFloat32,
                                          dot_float_type, mid_in));
  }

  uint32_t expert_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(define_ragged_dot(mid_in, mid_zp, mid_scale,
                                          weights[kDown], &expert_out));

  // 5. Combine: [R, D_out] gathered by the slots of the tokens
  // [T, K, 1] -> [T, K, D_out], weighted by the routing weights
  // [T, K, 1], reduced over K -> [T, D_out], and split back into
  // [B, N, D_out].
  uint32_t token_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_gather(subgraph, 1, &gather_axis_0, 3, expert_out,
                        dispatch.token_slot_id, &token_out, 0));
  uint32_t rw_flat = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_fuse_dim(subgraph, 0, 2, rw_scaled, &rw_flat, 0));
  const int32_t expand_axis_2 = 2;
  uint32_t rw_3d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 1, &expand_axis_2, rw_flat, &rw_3d, 0));
  uint32_t weighted_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
      subgraph, ynn_binary_multiply, token_out, rw_3d, &weighted_out, 0));
  const int32_t reduce_axis_1 = 1;
  uint32_t out_flat = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &reduce_axis_1,
                        weighted_out, YNN_INVALID_VALUE_ID, &out_flat, 0));
  const TfLiteTensor& output = context->tensors[node.outputs[0]];
  if (out_it != tensor_to_value_id.end()) {
    TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, kTfLiteFloat32,
                                          output.type, out_flat));
  }
  const size_t out_splits[] = {static_cast<size_t>(dispatch.batch_size), 0};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(subgraph, 0, 2, out_splits,
                                                 out_flat, &output_val, 0));

  // The reduction accumulates in fp32; an output value this node created
  // itself has to be narrowed to the type of the tensor it stands for.
  if (out_it == tensor_to_value_id.end()) {
    TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, kTfLiteFloat32,
                                          output.type, output_val));
  }

  tensor_to_value_id[node.outputs[0]] = output_val;

  return kTfLiteOk;
}

TfLiteStatus SetMoeDispatchShapes(TfLiteContext* context, ynn_runtime_t runtime,
                                  MoeDispatch& dispatch,
                                  const size_t (&expert_indices_dims)[3]) {
  const int data_tensor_index = dispatch.expert_indices_data_tensor_index >= 0
                                    ? dispatch.expert_indices_data_tensor_index
                                    : dispatch.expert_indices_tensor_index;
  const TfLiteTensor& expert_indices_data_tensor =
      context->tensors[data_tensor_index];
  const size_t batch_size = expert_indices_dims[0];
  const size_t seq_len = expert_indices_dims[1];
  const size_t top_k = expert_indices_dims[2];
  TF_LITE_ENSURE(context, batch_size > 0 && seq_len > 0 && top_k > 0);
  TF_LITE_ENSURE_EQ(context, batch_size,
                    static_cast<size_t>(dispatch.batch_size));
  if (dispatch.fixed_top_k > 0) {
    TF_LITE_ENSURE_EQ(context, top_k,
                      static_cast<size_t>(dispatch.fixed_top_k));
  }
  dispatch.top_k = static_cast<int>(top_k);
  TF_LITE_ENSURE(context,
                 seq_len <= std::numeric_limits<size_t>::max() / batch_size);
  const size_t num_tokens = batch_size * seq_len;
  TF_LITE_ENSURE(context,
                 top_k <= std::numeric_limits<size_t>::max() / num_tokens);
  const size_t num_selected =
      static_cast<size_t>(tflite::NumElements(&expert_indices_data_tensor));
  TF_LITE_ENSURE_EQ(context, num_selected, num_tokens * top_k);

  if (dispatch.external_tokens_tensor_index >= 0) {
    const TfLiteTensor& tokens =
        context->tensors[dispatch.external_tokens_tensor_index];
    TF_LITE_ENSURE(context, tokens.dims != nullptr && tokens.dims->size == 3);
    TF_LITE_ENSURE_EQ(context, static_cast<size_t>(tokens.dims->data[0]),
                      batch_size);
    TF_LITE_ENSURE_EQ(context, static_cast<size_t>(tokens.dims->data[1]),
                      seq_len);
  }
  if (dispatch.external_routing_weights_tensor_index >= 0) {
    const TfLiteTensor& rw =
        context->tensors[dispatch.external_routing_weights_tensor_index];
    TF_LITE_ENSURE(context, rw.dims != nullptr && rw.dims->size == 3);
    TF_LITE_ENSURE_EQ(context, static_cast<size_t>(rw.dims->data[0]),
                      batch_size);
    TF_LITE_ENSURE_EQ(context, static_cast<size_t>(rw.dims->data[1]), seq_len);
    TF_LITE_ENSURE_EQ(context, static_cast<size_t>(rw.dims->data[2]), top_k);
  }

  const size_t slot_dims[2] = {num_selected, 1};
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
      runtime, dispatch.slot_token_id, 2, slot_dims));
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
      runtime, dispatch.slot_expert_id, 2, slot_dims));
  const size_t token_slot_dims[3] = {num_tokens, top_k, 1};
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
      runtime, dispatch.token_slot_id, 3, token_slot_dims));
  return kTfLiteOk;
}

TfLiteStatus SetupMoeDispatch(TfLiteContext* context, ynn_runtime_t runtime,
                              MoeDispatch& dispatch) {
  const int data_tensor_index = dispatch.expert_indices_data_tensor_index >= 0
                                    ? dispatch.expert_indices_data_tensor_index
                                    : dispatch.expert_indices_tensor_index;
  const TfLiteTensor& expert_indices_data_tensor =
      context->tensors[data_tensor_index];
  TF_LITE_ENSURE_EQ(context, expert_indices_data_tensor.type, kTfLiteInt32);
  const int32_t* expert_indices_data = expert_indices_data_tensor.data.i32;
  TF_LITE_ENSURE(context, dispatch.top_k > 0);
  const size_t top_k = static_cast<size_t>(dispatch.top_k);
  const size_t num_selected =
      static_cast<size_t>(tflite::NumElements(&expert_indices_data_tensor));
  TF_LITE_ENSURE_EQ(context, num_selected % top_k, 0);
  const size_t num_tokens = num_selected / top_k;
  TF_LITE_ENSURE(context, num_selected == 0 || expert_indices_data != nullptr);
  const int num_experts = dispatch.num_experts;
  const size_t max_slots =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  TF_LITE_ENSURE(context, num_selected <= max_slots);

  // Count selections per expert and compute group offsets via prefix sum.
  dispatch.group_offsets.assign(num_experts, 0);
  for (size_t i = 0; i < num_selected; ++i) {
    const int32_t e = expert_indices_data[i];
    TF_LITE_ENSURE_MSG(context, 0 <= e && e < num_experts,
                       "MoE expert index out of range");
    ++dispatch.group_offsets[e];
  }

  int32_t offset = 0;
  for (int e = 0; e < num_experts; ++e) {
    const int32_t count = dispatch.group_offsets[e];
    dispatch.group_offsets[e] = offset;
    offset += count;
  }

  dispatch.slot_token.assign(num_selected, 0);
  dispatch.slot_expert.assign(num_selected, 0);
  dispatch.token_slot.resize(num_selected);

  for (size_t t = 0; t < num_tokens; ++t) {
    const int32_t* row = expert_indices_data + t * top_k;
    int32_t* out_slots = dispatch.token_slot.data() + t * top_k;
    for (size_t k = 0; k < top_k; ++k) {
      const int32_t e = row[k];
      const int32_t slot = dispatch.group_offsets[e]++;
      out_slots[k] = slot;
      dispatch.slot_token[slot] = static_cast<int32_t>(t);
      dispatch.slot_expert[slot] = e;
    }
  }

  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
      runtime, dispatch.slot_token_id, dispatch.slot_token.data()));
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
      runtime, dispatch.slot_expert_id, dispatch.slot_expert.data()));
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
      runtime, dispatch.token_slot_id, dispatch.token_slot.data()));
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
