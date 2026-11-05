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
#include <limits>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"
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
bool IsWeightTensorSupported(const TfLiteTensor& w, int scale_tensor_index) {
  if (w.quantization.type != kTfLiteMultiAxisQuantization) {
    return IsTensorSupported(w, /*allow_per_channel=*/true);
  }
  const auto* params =
      static_cast<const TfLiteMultiAxisQuantization*>(w.quantization.params);
  if (scale_tensor_index < 0 || params == nullptr) {
    return false;
  }
  // The scales have to be the explicit ones, with no zero points or blocks.
  if (params->scales != scale_tensor_index || params->zero_points >= 0 ||
      params->blocksize != 0) {
    return false;
  }
  // Per output channel and expert, i.e. axes 0 and 1 of [D_out, E, 1, D_in].
  const TfLiteIntArray* axes = params->quantized_dimensions;
  if (axes == nullptr || axes->size != 2 || axes->data[0] != 0 ||
      axes->data[1] != 1) {
    return false;
  }
  // A shallow copy with the quantization cleared, so that IsTensorSupported
  // only checks the type and dims.
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
                            const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE(context, IsMoe(registration, node));
  TF_LITE_ENSURE(context, node->inputs->size == kNumInputsWithoutWeightScales ||
                              node->inputs->size == kNumInputsWithWeightScales);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

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
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

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
    TF_LITE_ENSURE(context, IsWeightTensorSupported(*w, scale_tensor_index));
    TF_LITE_ENSURE(context, tflite::IsConstantTensor(w));
    TF_LITE_ENSURE(context,
                   IsFloatType(w->type) || IsQuantizedWeightType(w->type));
    TF_LITE_ENSURE_EQ(context, w->type, gate_weights.type);
    TF_LITE_ENSURE(context, w->dims != nullptr && w->dims->size == 4);
    if (has_weight_scales) {
      const TfLiteTensor& weight_scale = context->tensors[scale_tensor_index];
      TF_LITE_ENSURE(context, IsTensorSupported(weight_scale));
      TF_LITE_ENSURE(context, tflite::IsConstantTensor(&weight_scale));
      TF_LITE_ENSURE(context, IsFloatType(weight_scale.type));
    }
  }

  TF_LITE_ENSURE(context, tokens.dims != nullptr && tokens.dims->size == 3);
  TF_LITE_ENSURE(context, routing_weights.dims != nullptr &&
                              routing_weights.dims->size == 3);
  TF_LITE_ENSURE(context, expert_indices.dims != nullptr &&
                              expert_indices.dims->size == 3);
  TF_LITE_ENSURE(context, output.dims != nullptr && output.dims->size == 3);

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
                           uint32_t& next_external_id, MoeDispatch* dispatch,
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
  const int num_selected_experts =
      (expert_indices.dims && expert_indices.dims->size >= 1)
          ? expert_indices.dims->data[expert_indices.dims->size - 1]
          : 0;
  TF_LITE_ENSURE(context, num_selected_experts > 0);

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

  // 1. Transpose constant weights from 4D [*, E, 1, *] to 3D [E, *, *]
  // dropping the unit axis 2 so expert dimension E is on axis 0 and can be
  // gathered or batched over directly.
  int32_t perm_weights[] = {1, 3, 0};
  for (WeightValues& w : weights) {
    const uint32_t weights_val = GetOrCreateValueId(
        context, subgraph, tensor_to_value_id, w.tensor_index);
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, 3, perm_weights, weights_val, &w.value_id, 0));
  }

  if (is_quantized) {
    for (WeightValues& w : weights) {
      if (has_weight_scales) {
        const uint32_t weight_scale_val = GetOrCreateValueId(
            context, subgraph, tensor_to_value_id, w.scale_tensor_index);
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
            subgraph, 3, perm_weights, weight_scale_val, &w.scale_id, 0));
      } else {
        TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
            context, subgraph, context->tensors[w.tensor_index], &w.scale_id,
            &w.zero_point_id));
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

  int32_t gather_axis_0 = 0;

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

  // A dot against dynamically quantized activations has to carry both operands'
  // quantization parameters and dequantize the int32 accumulator; a float dot
  // is just a dot.
  auto define_dot = [&](uint32_t a_id, uint32_t a_zp, uint32_t a_scale,
                        const WeightValues& w,
                        uint32_t* out_id) -> TfLiteStatus {
    if (!is_quantized) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_id,
                                               w.value_id, YNN_INVALID_VALUE_ID,
                                               out_id, 0));
      return kTfLiteOk;
    }

    uint32_t dot_zp = YNN_INVALID_VALUE_ID;
    uint32_t dot_scale = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn::define_dot_quantization(
        subgraph, /*num_k_dims=*/1, a_id, a_zp, a_scale, w.value_id,
        w.zero_point_id, w.scale_id, dot_zp, dot_scale));

    // The dot accumulates into the negated zero point correction, so that the
    // raw accumulator is already zero-point corrected.
    uint32_t accum_init = YNN_INVALID_VALUE_ID;
    if (dot_zp != YNN_INVALID_VALUE_ID) {
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_unary(subgraph, ynn_unary_negate, dot_zp, &accum_init, 0));
    }

    uint32_t accum = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dot(
        subgraph, /*num_k_dims=*/1, a_id, w.value_id, accum_init, &accum, 0));

    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_dequantize(subgraph, accum, YNN_INVALID_VALUE_ID, dot_scale,
                              ynn_type_fp32, out_id, 0));
    return kTfLiteOk;
  };

  // The gate, gelu, up, product and down projection of the experts.
  auto define_experts = [&](uint32_t a, uint32_t a_zp, uint32_t a_scale,
                            uint32_t* out) -> TfLiteStatus {
    uint32_t gate = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(define_dot(a, a_zp, a_scale, weights[kGate], &gate));

    uint32_t act_gate = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn::define_approx_gelu(subgraph, gate, act_gate));

    uint32_t up = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(define_dot(a, a_zp, a_scale, weights[kUp], &up));

    uint32_t mid = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                act_gate, up, &mid, 0));

    uint32_t mid_in = mid;
    uint32_t mid_zp = YNN_INVALID_VALUE_ID;
    uint32_t mid_scale = YNN_INVALID_VALUE_ID;
    if (is_quantized) {
      mid_in = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(
          dynamic_quantize(mid, &mid_in, &mid_zp, &mid_scale));
    } else {
      // The dots accumulate in fp32, so the intermediate has to be narrowed
      // back to match the down weights.
      TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, kTfLiteFloat32,
                                            dot_float_type, mid_in));
    }

    return define_dot(mid_in, mid_zp, mid_scale, weights[kDown], out);
  };

  const int32_t expand_axis_3 = 3;
  // 2. Expert outputs for every (token, selected expert) pair:
  // -> [B, N, K, D_out]. Either lowering computes the same values.
  uint32_t out_4d = YNN_INVALID_VALUE_ID;
  if (dispatch != nullptr) {
    // The tokens are bucketed by expert on the host (see `SetupMoeDispatch`):
    // expert `e` computes the `C` slots `slot_token[e, :]`, where `C` is the
    // largest number of (token, selected expert) pairs routed to one expert.
    // This way each expert computes one contiguous run of rows.
    dispatch->expert_indices_tensor_index = node.inputs[idx.expert_indices];
    dispatch->num_experts = num_experts;
    auto define_external = [&](ynn_type type, size_t rank,
                               uint32_t* id) -> TfLiteStatus {
      *id = next_external_id++;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
          subgraph, type, rank, /*dims=*/nullptr, /*data=*/nullptr,
          YNN_VALUE_FLAG_EXTERNAL_INPUT, id));
      return kTfLiteOk;
    };
    TF_LITE_ENSURE_STATUS(
        define_external(ynn_type_int32, 2, &dispatch->slot_token_id));
    TF_LITE_ENSURE_STATUS(
        define_external(ynn_type_int32, 3, &dispatch->token_slot_id));

    const int32_t expand_axis_2 = 2;
    uint32_t slot_token_3d = YNN_INVALID_VALUE_ID;
    uint32_t token_slot_4d = YNN_INVALID_VALUE_ID;
    // [E, C] -> [E, C, 1]
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, 1, &expand_axis_2, dispatch->slot_token_id, &slot_token_3d,
        0));
    // [B, N, K] -> [B, N, K, 1]
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, 1, &expand_axis_3, dispatch->token_slot_id, &token_slot_4d,
        0));

    // The weights are converted while still constant, so the conversion is
    // folded once; the gather branch converts the gathered rows instead.
    if (!is_quantized) {
      for (WeightValues& w : weights) {
        TF_LITE_ENSURE_STATUS(ConvertIfNeeded(
            context, subgraph, context->tensors[w.tensor_index].type,
            dot_float_type, w.value_id));
      }
    }

    // [B, N, D_in] -> [T, D_in]
    uint32_t tokens_flat = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_fuse_dim(subgraph, 0, 2, tokens_val, &tokens_flat, 0));

    // Quantize the tokens before gathering them, so that each token is only
    // quantized once. [T, D_in] gathered by [E, C, 1] -> [E, C, D_in], and the
    // quantization parameters [T, 1] -> [E, C, 1].
    auto gather_slots = [&](uint32_t input, uint32_t* output) -> TfLiteStatus {
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
          subgraph, 1, &gather_axis_0, 3, input, slot_token_3d, output, 0));
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

    // [E, C, D_in] x [E, D_in, *] -> [E, C, *].
    uint32_t expert_out = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(define_experts(a, a_zp, a_scale, &expert_out));

    // [E, C, D_out] -> [E * C, D_out] gathered by the slots of the tokens
    // [B, N, K, 1] -> [B, N, K, D_out].
    uint32_t expert_out_flat = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_fuse_dim(subgraph, 0, 2, expert_out, &expert_out_flat, 0));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(subgraph, 1, &gather_axis_0, 4,
                                                expert_out_flat, token_slot_4d,
                                                &out_4d, 0));
  } else {
    // Expand tokens: [B, N, D_in] -> [B, N, 1, 1, D_in]
    uint32_t tokens_5d = YNN_INVALID_VALUE_ID;
    int32_t expand_axes_2_3[] = {2, 3};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, 2, expand_axes_2_3, tokens_val, &tokens_5d, 0));

    if (!is_quantized) {
      TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, tokens.type,
                                            dot_float_type, tokens_5d));
    }

    // Expand expert indices: [B, N, K] -> [B, N, K, 1, 1]
    uint32_t ei_5d = YNN_INVALID_VALUE_ID;
    int32_t expand_axes_3_4[] = {3, 4};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, 2, expand_axes_3_4, expert_indices_val, &ei_5d, 0));

    // Gather weights along axis 0 (expert dim E) for active choices:
    // [E, D_in, D_mid] gathered by [B, N, K, 1, 1] -> [B, N, K, D_in, D_mid]
    for (WeightValues& w : weights) {
      uint32_t gathered = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
          subgraph, 1, &gather_axis_0, 5, w.value_id, ei_5d, &gathered, 0));
      w.value_id = gathered;
    }

    if (!is_quantized) {
      for (WeightValues& w : weights) {
        TF_LITE_ENSURE_STATUS(ConvertIfNeeded(
            context, subgraph, context->tensors[w.tensor_index].type,
            dot_float_type, w.value_id));
      }
    }

    if (is_quantized && has_weight_scales) {
      for (WeightValues& w : weights) {
        uint32_t gathered = YNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
            subgraph, 1, &gather_axis_0, 5, w.scale_id, ei_5d, &gathered, 0));
        w.scale_id = gathered;
      }
    }

    uint32_t tokens_in = tokens_5d;
    uint32_t tokens_zp = YNN_INVALID_VALUE_ID;
    uint32_t tokens_scale = YNN_INVALID_VALUE_ID;
    if (is_quantized) {
      tokens_in = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(
          dynamic_quantize(tokens_5d, &tokens_in, &tokens_zp, &tokens_scale));
    }

    // Batched dot products: [B, N, 1, 1, D_in] x [B, N, K, D_in, *] ->
    // [B, N, K, 1, D_out].
    uint32_t out_5d = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(
        define_experts(tokens_in, tokens_zp, tokens_scale, &out_5d));

    // Squeeze axis 3: [B, N, K, 1, D_out] -> [B, N, K, D_out]
    // TODO: b/558433816 - This should be static_transpose, but it is faster to
    // use a fuse_dim due to scheduling issues.
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_fuse_dim(subgraph, 2, 2, out_5d, &out_4d, 0));
  }

  // 3. Routing weights & scale: [B, N, K] -> [B, N, K, 1]
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

  uint32_t rw_4d = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 1, &expand_axis_3, rw_scaled, &rw_4d, 0));

  // 4. Multiply weighted outputs: [B, N, K, D_out] * [B, N, K, 1] -> [B, N, K,
  // D_out]
  uint32_t weighted_out = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                              out_4d, rw_4d, &weighted_out, 0));

  // 5. Reduce sum over K (axis 2): [B, N, K, D_out] -> [B, N, D_out]
  const int32_t reduce_axis_2 = 2;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, &reduce_axis_2,
                        weighted_out, YNN_INVALID_VALUE_ID, &output_val, 0));

  // The reduction accumulates in fp32; an output value this node created
  // itself has to be narrowed to the type of the tensor it stands for.
  const TfLiteTensor& output = context->tensors[node.outputs[0]];
  if (out_it == tensor_to_value_id.end()) {
    TF_LITE_ENSURE_STATUS(ConvertIfNeeded(context, subgraph, kTfLiteFloat32,
                                          output.type, output_val));
  }

  tensor_to_value_id[node.outputs[0]] = output_val;

  return kTfLiteOk;
}

bool ShouldDispatchMoeOnHost(TfLiteContext* context, const NodeInfo& node,
                             const std::vector<int>& partition_inputs,
                             const TfLiteYNNPackDelegateOptions& options) {
  if (!options.dispatch_moe_on_host) {
    return false;
  }
  const bool has_weight_scales =
      node.inputs.size() == kNumInputsWithWeightScales;
  const MoeInputIndices idx = GetInputIndices(has_weight_scales);
  const int expert_indices_tensor_index = node.inputs[idx.expert_indices];
  if (!absl::c_linear_search(partition_inputs, expert_indices_tensor_index)) {
    return false;
  }
  const TfLiteTensor& expert_indices =
      context->tensors[expert_indices_tensor_index];
  return expert_indices.dims != nullptr && expert_indices.dims->size == 3 &&
         expert_indices.dims->data[0] * expert_indices.dims->data[1] > 1;
}

TfLiteStatus SetMoeDispatchShapes(TfLiteContext* context, ynn_runtime_t runtime,
                                  const MoeDispatch& dispatch) {
  const TfLiteTensor& expert_indices =
      context->tensors[dispatch.expert_indices_tensor_index];
  TF_LITE_ENSURE(context, expert_indices.dims != nullptr &&
                              expert_indices.dims->size == 3);
  const size_t slots_dims[] = {static_cast<size_t>(dispatch.num_experts),
                               dispatch.capacity};
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
      runtime, dispatch.slot_token_id, 2, slots_dims));
  size_t token_slot_dims[3];
  std::copy_n(expert_indices.dims->data, 3, token_slot_dims);
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
      runtime, dispatch.token_slot_id, 3, token_slot_dims));
  return kTfLiteOk;
}

TfLiteStatus SetupMoeDispatch(TfLiteContext* context, ynn_runtime_t runtime,
                              MoeDispatch& dispatch, bool* reshape_required) {
  const TfLiteTensor& expert_indices =
      context->tensors[dispatch.expert_indices_tensor_index];
  TF_LITE_ENSURE_EQ(context, expert_indices.type, kTfLiteInt32);
  TF_LITE_ENSURE(context, expert_indices.dims != nullptr &&
                              expert_indices.dims->size == 3);
  const int32_t* expert_indices_data = expert_indices.data.i32;
  const size_t num_selected = tflite::NumElements(&expert_indices);
  const size_t top_k = expert_indices.dims->data[2];
  // `DefineMoeNode` checked this when the runtime was built, but in
  // dynamic-shape mode a resize to K = 0 does not rebuild it.
  TF_LITE_ENSURE(context, top_k > 0);
  TF_LITE_ENSURE(context, expert_indices_data != nullptr);
  const int num_experts = dispatch.num_experts;

  // Counting sort of the (token, selected expert) pairs by expert.
  dispatch.count.assign(num_experts, 0);
  for (size_t i = 0; i < num_selected; ++i) {
    const int32_t e = expert_indices_data[i];
    TF_LITE_ENSURE_MSG(context, 0 <= e && e < num_experts,
                       "MoE expert index out of range");
    ++dispatch.count[e];
  }
  // A zero capacity would make the tensors empty; keep one padding slot.
  size_t capacity = 1;
  for (int32_t c : dispatch.count) {
    capacity = std::max<size_t>(capacity, c);
  }
  const size_t max_slots =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  TF_LITE_ENSURE(context,
                 capacity <= max_slots / static_cast<size_t>(num_experts));
  const size_t num_slots = static_cast<size_t>(num_experts) * capacity;

  // Changing the capacity changes the shapes of the whole expert computation,
  // so the runtime has to be reshaped. We use the exact capacity rather than
  // rounding it up to avoid reshaping: reshaping is cheap compared to
  // computing padding slots.
  if (capacity != dispatch.capacity) {
    dispatch.capacity = capacity;
    TF_LITE_ENSURE_STATUS(SetMoeDispatchShapes(context, runtime, dispatch));
    *reshape_required = true;
  }

  // The padding slots compute token 0 (any valid token would do); their
  // outputs are never gathered by `token_slot`.
  dispatch.slot_token.assign(num_slots, 0);
  dispatch.token_slot.resize(num_selected);
  std::fill(dispatch.count.begin(), dispatch.count.end(), 0);
  for (size_t i = 0; i < num_selected; ++i) {
    const int32_t e = expert_indices_data[i];
    const size_t slot = e * capacity + dispatch.count[e]++;
    dispatch.slot_token[slot] = static_cast<int32_t>(i / top_k);
    dispatch.token_slot[i] = static_cast<int32_t>(slot);
  }
  return kTfLiteOk;
}

TfLiteStatus SetMoeDispatchData(TfLiteContext* context, ynn_runtime_t runtime,
                                MoeDispatch& dispatch) {
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
      runtime, dispatch.slot_token_id, dispatch.slot_token.data()));
  TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
      runtime, dispatch.token_slot_id, dispatch.token_slot.data()));
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
