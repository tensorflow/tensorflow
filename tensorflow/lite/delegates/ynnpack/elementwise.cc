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

#include "tensorflow/lite/delegates/ynnpack/elementwise.h"

#include <cstdint>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

// ==============================================================================
// Unary Operations
// ==============================================================================

TfLiteStatus IsUnaryOpSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node,
                                TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  int builtin_code = registration->builtin_code;
  ynn_unary_operator op = GetYnnUnaryOperator(builtin_code);
  bool is_decomposed = (builtin_code == kTfLiteBuiltinGelu ||
                        builtin_code == kTfLiteBuiltinElu ||
                        builtin_code == kTfLiteBuiltinLeakyRelu ||
                        builtin_code == kTfLiteBuiltinHardSwish ||
                        builtin_code == kTfLiteBuiltinRelu ||
                        builtin_code == kTfLiteBuiltinRelu6 ||
                        builtin_code == kTfLiteBuiltinReluN1To1);
  TF_LITE_ENSURE(context, op != ynn_unary_invalid || is_decomposed);

  if (op != ynn_unary_convert) {
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
  }

  if (op == ynn_unary_convert) {
    // Reject constant inputs to allow TFLite caching optimization.
    TF_LITE_ENSURE_MSG(context, input.allocation_type != kTfLiteMmapRo,
                       "Constant input for convert is not supported");
    // YNNPACK convert to integer uses rounding, but TFLite Cast expects
    // truncation. Reject all float-to-integer conversions.
    if ((input.type == kTfLiteFloat32 || input.type == kTfLiteFloat16) &&
        (output.type == kTfLiteInt32 || output.type == kTfLiteInt8 ||
         output.type == kTfLiteUInt8 || output.type == kTfLiteInt16)) {
      TF_LITE_ENSURE_MSG(context, false,
                         "Float to integer conversion is not supported");
    }
  }

  if (op == ynn_unary_square_root || op == ynn_unary_reciprocal_square_root) {
    // Reject quantized Sqrt/Rsqrt because we cannot report errors for negative
    // inputs during execution if we delegate them.
    TF_LITE_ENSURE_MSG(context,
                       input.quantization.type == kTfLiteNoQuantization,
                       "Quantized Sqrt/Rsqrt is not supported");
  }

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  // Check for fused activation.
  TfLiteFusedActivation activation = GetFusedActivation(registration, node);
  TF_LITE_ENSURE(context, IsActivationSupported(activation, output.type));

  return kTfLiteOk;
}

TfLiteStatus DefineUnaryNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TensorToValueIdMap& tensor_to_value_id,
                             const NodeInfo& node) {
  return DefineDecomposedUnaryNode(
      context, subgraph, tensor_to_value_id, node,
      [context, subgraph, &node](uint32_t input_id,
                                 uint32_t& output_id) -> TfLiteStatus {
        ynn_unary_operator op = GetYnnUnaryOperator(node.builtin_code);
        if (op != ynn_unary_invalid) {
          TF_LITE_ENSURE_YNN_STATUS(
              ynn_define_unary(subgraph, op, input_id, &output_id, 0));
          return kTfLiteOk;
        }

        switch (node.builtin_code) {
          case kTfLiteBuiltinGelu: {
            TfLiteNode* tflite_node;
            TfLiteRegistration* reg;
            TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
                context, node.node_index, &tflite_node, &reg));
            const auto* params =
                static_cast<const TfLiteGeluParams*>(tflite_node->builtin_data);
            bool approximate = params && params->approximate;
            if (approximate) {
              TF_LITE_ENSURE_YNN_STATUS(
                  ynn::define_approx_gelu(subgraph, input_id, output_id));
            } else {
              TF_LITE_ENSURE_YNN_STATUS(
                  ynn::define_gelu(subgraph, input_id, output_id));
            }
            return kTfLiteOk;
          }
          case kTfLiteBuiltinElu:
            TF_LITE_ENSURE_YNN_STATUS(
                ynn::define_elu(subgraph, input_id, 1.0f, output_id));
            return kTfLiteOk;
          case kTfLiteBuiltinLeakyRelu: {
            TfLiteNode* tflite_node;
            TfLiteRegistration* reg;
            TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
                context, node.node_index, &tflite_node, &reg));
            const auto* params = static_cast<const TfLiteLeakyReluParams*>(
                tflite_node->builtin_data);
            float alpha = params ? params->alpha : 0.2f;
            TF_LITE_ENSURE_YNN_STATUS(
                ynn::define_leaky_relu(subgraph, input_id, alpha, output_id));
            return kTfLiteOk;
          }
          case kTfLiteBuiltinHardSwish:
            TF_LITE_ENSURE_YNN_STATUS(
                ynn::define_hardswish(subgraph, input_id, output_id));
            return kTfLiteOk;
          case kTfLiteBuiltinRelu:
          case kTfLiteBuiltinRelu6:
          case kTfLiteBuiltinReluN1To1: {
            TfLiteFusedActivation activation = kTfLiteActNone;
            if (node.builtin_code == kTfLiteBuiltinRelu) {
              activation = kTfLiteActRelu;
            } else if (node.builtin_code == kTfLiteBuiltinRelu6) {
              activation = kTfLiteActRelu6;
            } else if (node.builtin_code == kTfLiteBuiltinReluN1To1) {
              activation = kTfLiteActReluN1To1;
            }
            return ApplyActivation(context, subgraph, activation, input_id,
                                   output_id, node.outputs[0], ynn_type_fp32);
          }
          default:
            TF_LITE_ENSURE(context, false);
        }
      });
}

// ==============================================================================
// Binary Operations
// ==============================================================================

TfLiteStatus IsBinaryOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input1 = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& input2 = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input1_ynn_type = GetYnnType(input1.type);
  ynn_type input2_ynn_type = GetYnnType(input2.type);
  ynn_type output_ynn_type = GetYnnType(output.type);

  switch (input1_ynn_type) {
    case ynn_type_int8:
    case ynn_type_uint8:
    case ynn_type_int32:
      return kTfLiteError;
    default:
      break;
  }

  TF_LITE_ENSURE(context, input1_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, input2_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input1.type, output.type);
  TF_LITE_ENSURE_EQ(context, input2.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input1));
  TF_LITE_ENSURE(context, IsSupportedQuantization(input2));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  // YNNPACK integer division is floor division, but TFLite expects truncation.
  if ((registration->builtin_code == kTfLiteBuiltinDiv ||
       registration->builtin_code == kTfLiteBuiltinStablehloDivide) &&
      output.type == kTfLiteInt32) {
    TF_LITE_ENSURE_MSG(
        context, false,
        "Integer division is not supported (truncation mismatch)");
  }

  // Check for fused activation.
  TfLiteFusedActivation activation = GetFusedActivation(registration, node);
  TF_LITE_ENSURE(context, IsActivationSupported(activation, output.type));

  return kTfLiteOk;
}

TfLiteStatus DefineBinaryNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                              TensorToValueIdMap& tensor_to_value_id,
                              const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);
  int input1_tensor_index = node.inputs[0];
  int input2_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  uint32_t input1_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input1_tensor_index);
  uint32_t input2_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input2_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input1_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, input2_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  uint32_t float_input1_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(
      context, subgraph, tensor_to_value_id, input1_tensor_index, input1_val_id,
      &float_input1_val_id));

  uint32_t float_input2_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(
      context, subgraph, tensor_to_value_id, input2_tensor_index, input2_val_id,
      &float_input2_val_id));

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  ynn_type internal_type = IsQuantized(output_tensor)
                               ? ynn_type_fp32
                               : GetYnnType(output_tensor.type);
  TfLiteFusedActivation activation = node.activation;
  bool is_output_quantized = IsQuantized(output_tensor);

  uint32_t float_output_val_id =
      !is_output_quantized && activation == kTfLiteActNone
          ? output_val_id
          : YNN_INVALID_VALUE_ID;

  uint32_t broadcasted_input1_val_id = float_input1_val_id;
  uint32_t broadcasted_input2_val_id = float_input2_val_id;

  const TfLiteTensor& input1_tensor = context->tensors[input1_tensor_index];
  const TfLiteTensor& input2_tensor = context->tensors[input2_tensor_index];
  int rank1 = input1_tensor.dims->size;
  int rank2 = input2_tensor.dims->size;

  if (!IsStablehloOp(node.builtin_code)) {
    // If the op is not a StableHLO op, we might need to broadcast.
    TF_LITE_ENSURE_STATUS(ImplementMutualBroadcasting(
        context, subgraph, rank1, rank2, 0, 0, broadcasted_input1_val_id,
        broadcasted_input2_val_id));
  }

  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, GetYnnBinaryOperator(node.builtin_code),
                        broadcasted_input1_val_id, broadcasted_input2_val_id,
                        &float_output_val_id, 0));

  uint32_t active_output_val_id = float_output_val_id;
  if (activation != kTfLiteActNone) {
    uint32_t activation_output_val_id =
        is_output_quantized ? YNN_INVALID_VALUE_ID : output_val_id;
    TF_LITE_ENSURE_STATUS(ApplyActivation(
        context, subgraph, activation, float_output_val_id,
        activation_output_val_id, output_tensor_index, internal_type));
    active_output_val_id = activation_output_val_id;
  }

  if (is_output_quantized) {
    TF_LITE_ENSURE_STATUS(Quantize(context, subgraph, tensor_to_value_id,
                                   output_tensor_index, active_output_val_id,
                                   output_val_id));
  }

  return kTfLiteOk;
}

// ==============================================================================
// Ternary Operations (Clamp)
// ==============================================================================

TfLiteStatus IsStablehloClampSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& min_tensor = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& operand_tensor = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& max_tensor = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type min_ynn_type = GetYnnType(min_tensor.type);
  ynn_type operand_ynn_type = GetYnnType(operand_tensor.type);
  ynn_type max_ynn_type = GetYnnType(max_tensor.type);
  ynn_type output_ynn_type = GetYnnType(output.type);

  switch (operand_ynn_type) {
    case ynn_type_int8:
    case ynn_type_uint8:
    case ynn_type_int32:
    case ynn_type_fp16:
    case ynn_type_bf16:
      return kTfLiteError;
    default:
      break;
  }

  TF_LITE_ENSURE(context, min_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, operand_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, max_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, min_tensor.type, output.type);
  TF_LITE_ENSURE_EQ(context, operand_tensor.type, output.type);
  TF_LITE_ENSURE_EQ(context, max_tensor.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(min_tensor));
  TF_LITE_ENSURE(context, IsSupportedQuantization(operand_tensor));
  TF_LITE_ENSURE(context, IsSupportedQuantization(max_tensor));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  // Fused activation is not supported for StableHLO Clamp.
  TfLiteFusedActivation activation = GetFusedActivation(registration, node);
  TF_LITE_ENSURE_EQ(context, activation, kTfLiteActNone);

  return kTfLiteOk;
}

TfLiteStatus DefineStablehloClampNode(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      TensorToValueIdMap& tensor_to_value_id,
                                      const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 3);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int min_tensor_index = node.inputs[0];
  int operand_tensor_index = node.inputs[1];
  int max_tensor_index = node.inputs[2];
  int output_tensor_index = node.outputs[0];

  uint32_t min_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, min_tensor_index);
  uint32_t operand_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, operand_tensor_index);
  uint32_t max_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, max_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, min_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, operand_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, max_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  uint32_t float_min_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(context, subgraph,
                                           tensor_to_value_id, min_tensor_index,
                                           min_val_id, &float_min_val_id));

  uint32_t float_operand_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(
      context, subgraph, tensor_to_value_id, operand_tensor_index,
      operand_val_id, &float_operand_val_id));

  uint32_t float_max_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(context, subgraph,
                                           tensor_to_value_id, max_tensor_index,
                                           max_val_id, &float_max_val_id));

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  bool is_output_quantized = IsQuantized(output_tensor);

  uint32_t float_output_val_id =
      !is_output_quantized ? output_val_id : YNN_INVALID_VALUE_ID;

  // temp = max(operand, min)
  uint32_t temp_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_max, float_operand_val_id,
                        float_min_val_id, &temp_val_id, 0));

  // output = min(temp, max)
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_min,
                                              temp_val_id, float_max_val_id,
                                              &float_output_val_id, 0));

  if (is_output_quantized) {
    TF_LITE_ENSURE_STATUS(Quantize(context, subgraph, tensor_to_value_id,
                                   output_tensor_index, float_output_val_id,
                                   output_val_id));
  }

  return kTfLiteOk;
}

// ==============================================================================
// Quantize/Dequantize Operations
// ==============================================================================

TfLiteStatus IsQuantizeSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, !IsQuantized(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  return kTfLiteOk;
}

TfLiteStatus DefineQuantizeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                                TensorToValueIdMap& tensor_to_value_id,
                                const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 1);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, output_tensor, &scale_id, &zp_id));

  ynn_type ynn_type = GetYnnType(output_tensor.type);
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(
      subgraph, input_val_id, ynn_type, zp_id, scale_id, &output_val_id, 0));
  return kTfLiteOk;
}

TfLiteStatus IsDequantizeSupported(const TfLiteRegistration* registration,
                                   const TfLiteNode* node,
                                   TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, !IsQuantized(output));

  return kTfLiteOk;
}

TfLiteStatus DefineDequantizeNode(TfLiteContext* context,
                                  ynn_subgraph_t subgraph,
                                  TensorToValueIdMap& tensor_to_value_id,
                                  const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 1);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
      context, subgraph, input_tensor, &scale_id, &zp_id));

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_dequantize(subgraph, input_val_id, zp_id,
                                                  scale_id, ynn_type_fp32,
                                                  &output_val_id, 0));
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
