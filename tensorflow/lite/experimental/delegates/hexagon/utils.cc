/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/delegates/hexagon/utils.h"

#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace {

bool IsActivationReluOrNone(TfLiteFusedActivation activation) {
  return (activation == kTfLiteActRelu || activation == kTfLiteActRelu6 ||
          activation == kTfLiteActRelu1 || activation == kTfLiteActNone);
}

bool TensorTypeMatch(int tensor_id, TfLiteContext* context,
                     TfLiteType tensor_type) {
  const auto& tensor = context->tensors[tensor_id];
  return tensor.type == tensor_type;
}

// For each input tensor i, checks if the type matches one of the possibilities
// in per_input_possible_types[i].
bool InputsWithCorrectTypes(
    const TfLiteNode* node, TfLiteContext* context,
    const std::vector<std::vector<TfLiteType>>& per_input_possible_types) {
  if (node->inputs->size != per_input_possible_types.size()) return false;
  for (int i = 0; i < per_input_possible_types.size(); ++i) {
    // Skip optional tensor.
    if (node->inputs->data[i] == -1) continue;
    bool type_found = false;
    for (auto possible_type : per_input_possible_types[i]) {
      if (TensorTypeMatch(node->inputs->data[i], context, possible_type)) {
        type_found = true;
        break;
      }
    }
    if (!type_found) return false;
  }
  return true;
}

}  // namespace

TfLiteStatus Get4DShape(unsigned int* batch_size, unsigned int* height_size,
                        unsigned int* width_size, unsigned int* depth_size,
                        TfLiteIntArray* dims) {
  if (dims->size > 4) return kTfLiteError;
  unsigned int* dim[] = {batch_size, height_size, width_size, depth_size};
  for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
  for (int i = 4 - dims->size; i < 4; ++i) {
    *dim[i] = dims->data[i - (4 - dims->size)];
  }
  return kTfLiteOk;
}

// We maintain an op-version whitelist here to ensure we don't accept unintended
// ops.
bool CheckOpVersion(const TfLiteRegistration* registration) {
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMirrorPad:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinPack:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinQuantize:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinSplit:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTranspose:
    case kTfLiteBuiltinTransposeConv:
      return registration->version <= 2;
    case kTfLiteBuiltinRelu:
      return registration->version == 2;
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
      return registration->version <= 3;
    case kTfLiteBuiltinFullyConnected:
      return registration->version <= 4;
    default:
      return registration->version == 1;
  }
}

bool IsNodeSupportedByHexagon(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
  // Ensure all inputs & outputs have dim <= 4.
  int tensor_id;
  for (int i = 0; i < node->inputs->size; ++i) {
    tensor_id = node->inputs->data[i];
    // Skip optional tensors. Builders should handle optional tensors
    // not available.
    if (tensor_id == -1) continue;
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    tensor_id = node->outputs->data[i];
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }

  if (!CheckOpVersion(registration)) return false;

  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteAddParams* add_params =
          reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
      return IsActivationReluOrNone(add_params->activation);
    }
    case kTfLiteBuiltinMul: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteMulParams* mul_params =
          reinterpret_cast<const TfLiteMulParams*>(node->builtin_data);
      // TODO(b/129276536): Add support for activation on Mul node.
      return IsActivationReluOrNone(mul_params->activation);
    }
    case kTfLiteBuiltinSub: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteSubParams* sub_params =
          reinterpret_cast<const TfLiteSubParams*>(node->builtin_data);
      return IsActivationReluOrNone(sub_params->activation);
    }
    case kTfLiteBuiltinSum:
      // TODO(b/139277813): Enable these when they pass unit tests. These seem
      // to recompute the output min/max instead of taking them as inputs, which
      // causes an unexpected shift in dequantized values.
      return false;
    case kTfLiteBuiltinMean: {
      return InputsWithCorrectTypes(
                 node, context,
                 {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
             IsConstantTensor(GetInput(context, node, 1));
    }
    case kTfLiteBuiltinMirrorPad: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) ||
          !IsConstantTensor(GetInput(context, node, 1)))
        return false;
      const TfLiteMirrorPaddingParams* params =
          reinterpret_cast<const TfLiteMirrorPaddingParams*>(
              node->builtin_data);
      return params->mode == kTfLiteMirrorPaddingReflect ||
             params->mode == kTfLiteMirrorPaddingSymmetric;
    }
    case kTfLiteBuiltinPad: {
      // TODO(b/139277813): Currently we only support padding with the default
      // of 0. Add support for user-defined constant if required.
      return (
          node->inputs->size == 2 &&
          InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
          IsConstantTensor(GetInput(context, node, 1)));
    }
    case kTfLiteBuiltinFullyConnected: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32, kTfLiteNoType}})) {
        return false;
      }

      bool bias_const_or_no_bias = true;
      if (node->inputs->data[2] != -1) {
        const auto& bias_tensor = context->tensors[node->inputs->data[2]];
        bias_const_or_no_bias = bias_tensor.allocation_type == kTfLiteMmapRo;
      }

      const TfLiteFullyConnectedParams* matmul_params =
          reinterpret_cast<const TfLiteFullyConnectedParams*>(
              node->builtin_data);
      return (bias_const_or_no_bias &&
              IsActivationReluOrNone(matmul_params->activation) &&
              matmul_params->keep_num_dims == false &&
              matmul_params->weights_format ==
                  kTfLiteFullyConnectedWeightsFormatDefault);
    }
    case kTfLiteBuiltinConcatenation: {
      // All concatenated tensors must be 8-bit.
      for (int i = 0; i < node->inputs->size; ++i) {
        if (!TensorTypeMatch(node->inputs->data[i], context, kTfLiteUInt8) &&
            !TensorTypeMatch(node->inputs->data[i], context, kTfLiteInt8))
          return false;
      }
      return true;
    }
    case kTfLiteBuiltinMaxPool2d: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      // TODO(b/129276536): Add support for activation here.
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return pool_params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinAveragePool2d: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return (node->inputs->size == 1 &&
              pool_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinTransposeConv: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteInt32},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteTransposeConvParams* params =
          reinterpret_cast<const TfLiteTransposeConvParams*>(
              node->builtin_data);
      return (params->stride_height <= 3 && params->stride_width <= 3 &&
              (params->padding == kTfLitePaddingSame ||
               params->padding == kTfLitePaddingValid));
    }
    case kTfLiteBuiltinConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32}}))
        return false;
      const TfLiteConvParams* conv_params =
          reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 &&
              conv_params->dilation_height_factor == 1 &&
              conv_params->dilation_width_factor == 1);
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32}}))
        return false;

      // Check dilation.
      const TfLiteDepthwiseConvParams* conv_params =
          reinterpret_cast<const TfLiteDepthwiseConvParams*>(
              node->builtin_data);
      const bool dilation = conv_params->dilation_height_factor != 1 ||
                            conv_params->dilation_width_factor != 1;
      if (dilation) {
        // We only support dilations when stride == 1.
        if (conv_params->stride_height != 1 || conv_params->stride_width != 1)
          return false;
      }

      // We currently only support depth_multiplier > 1 when:
      // 1. dilation_factor == 1 AND
      // 2. input_depth == 1
      // TODO(b/143759564): Add support for general case.
      const auto& input = context->tensors[node->inputs->data[0]];
      const bool supported_depth_multiplier =
          conv_params->depth_multiplier == 1 ||
          (!dilation && input.dims->size == 4 && input.dims->data[3] == 1);

      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 && supported_depth_multiplier);
    }
    case kTfLiteBuiltinReshape: {
      if (node->inputs->size > 2 ||
          (!TensorTypeMatch(node->inputs->data[0], context, kTfLiteUInt8) &&
           !TensorTypeMatch(node->inputs->data[0], context, kTfLiteInt8)))
        return false;
      return true;
    }
    case kTfLiteBuiltinSoftmax: {
      return (
          InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}));
    }
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinLogistic: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinResizeNearestNeighbor: {
      return InputsWithCorrectTypes(
                 node, context,
                 {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
             IsConstantTensor(GetInput(context, node, 1));
    }
    case kTfLiteBuiltinL2Normalization: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteL2NormParams* norm_params =
          reinterpret_cast<const TfLiteL2NormParams*>(node->builtin_data);
      return (norm_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
      return InputsWithCorrectTypes(
          node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}});
    case kTfLiteBuiltinSplit: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteInt32}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const auto& input_tensor = context->tensors[node->inputs->data[1]];
      const bool is_four_dim_or_less = input_tensor.dims->size < 5;
      // We need splitting axis to be constant, so Hexagon knows output
      // shapes.
      return is_four_dim_or_less &&
             IsConstantTensor(GetInput(context, node, 0));
    }
    case kTfLiteBuiltinResizeBilinear: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) ||
          !IsConstantTensor(GetInput(context, node, 1))) {
        return false;
      }
      const auto& size_tensor = context->tensors[node->inputs->data[1]];
      // TODO(b/143105433): Latency increase significantly with large size
      // value. Limiting to 65 for now.
      return NumElements(&size_tensor) == 2 && size_tensor.data.i32[0] < 66 &&
             size_tensor.data.i32[1] < 66;
    }
    case kTfLiteBuiltinNeg: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinTranspose: {
      return InputsWithCorrectTypes(
          node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}});
    }
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinQuantize: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinMinimum: {
      return InputsWithCorrectTypes(
          node, context,
          {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinMaximum: {
      return InputsWithCorrectTypes(
          node, context,
          {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinSlice: {
      const auto& begins_tensor = context->tensors[node->inputs->data[1]];
      const auto& sizes_tensor = context->tensors[node->inputs->data[2]];
      if (!IsConstantTensor(&begins_tensor) || !IsConstantTensor(&sizes_tensor))
        return false;
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8},
                                     {kTfLiteInt32, kTfLiteInt64},
                                     {kTfLiteInt32, kTfLiteInt64}});
    }
    case kTfLiteBuiltinPack: {
      // All tensors must be 8-bit.
      for (int i = 0; i < node->inputs->size; ++i) {
        if (!TensorTypeMatch(node->inputs->data[i], context, kTfLiteUInt8) &&
            !TensorTypeMatch(node->inputs->data[i], context, kTfLiteInt8))
          return false;
      }
      return true;
    }
    case kTfLiteBuiltinStridedSlice: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32},
                                   {kTfLiteInt32},
                                   {kTfLiteInt32}}))
        return false;
      const auto& begins_tensor = context->tensors[node->inputs->data[1]];
      const auto& ends_tensor = context->tensors[node->inputs->data[2]];
      const auto& step_tensor = context->tensors[node->inputs->data[3]];
      if (!IsConstantTensor(&begins_tensor) ||
          !IsConstantTensor(&ends_tensor) || !IsConstantTensor(&step_tensor))
        return false;
      const TfLiteStridedSliceParams* params =
          reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
      // Hexagon doesn't support ellipsis/new-axis masks.
      return (params->ellipsis_mask == 0 && params->new_axis_mask == 0);
    }
    default:
      return false;
  }
  return false;
}

}  // namespace tflite
