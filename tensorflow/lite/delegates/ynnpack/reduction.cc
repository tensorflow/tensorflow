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

#include "tensorflow/lite/delegates/ynnpack/reduction.h"

#include <cstdint>
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

TfLiteStatus IsReductionSupported(const TfLiteRegistration* registration,
                                  const TfLiteNode* node,
                                  TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  TF_LITE_ENSURE(context, NumElements(&input) > 0);
  const TfLiteTensor& axes = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE(context, axes.allocation_type == kTfLiteMmapRo);
  TF_LITE_ENSURE_EQ(context, axes.type, kTfLiteInt32);

  // YNNPACK reduce only supports 0D or 1D axes for now.
  TF_LITE_ENSURE(context, axes.dims->size <= 1);
  TF_LITE_ENSURE_MSG(context, input.dims->size <= YNN_MAX_TENSOR_RANK,
                     "Input rank exceeds max rank");

  return kTfLiteOk;
}

TfLiteStatus DefineReductionNode(TfLiteContext* context,
                                 ynn_subgraph_t subgraph,
                                 TensorToValueIdMap& tensor_to_value_id,
                                 const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int axes_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& axes_tensor = context->tensors[axes_tensor_index];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  int num_axes = axes_tensor.dims->size == 0 ? 1 : axes_tensor.dims->data[0];
  const int32_t* axes_data =
      reinterpret_cast<const int32_t*>(axes_tensor.data.raw);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteReducerParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  bool is_quantized = IsQuantized(input_tensor);
  bool is_mean = (node.builtin_code == kTfLiteBuiltinMean);
  bool is_sum = (node.builtin_code == kTfLiteBuiltinSum);

  if (is_mean || is_sum) {
    const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
    ynn_type output_type = GetYnnType(output_tensor.type);

    uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t input_zp_id = YNN_INVALID_VALUE_ID;
    if (is_quantized) {
      TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
          context, subgraph, input_tensor, &input_scale_id, &input_zp_id));
    }

    uint32_t output_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t output_zp_id = YNN_INVALID_VALUE_ID;
    if (is_quantized) {
      TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
          context, subgraph, output_tensor, &output_scale_id, &output_zp_id));
    }

    TF_LITE_ENSURE_YNN_STATUS(ynn::define_reduce_sum(
        subgraph, num_axes, axes_data, input_val_id, input_zp_id,
        input_scale_id, params->keep_dims, is_mean, /*squared=*/false,
        output_type, output_zp_id, output_scale_id, output_val_id));
  } else {
    // For Min/Max, we can reduce directly (same for float and quantized).
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
        subgraph, GetYnnReduceOperator(node.builtin_code), num_axes, axes_data,
        input_val_id, YNN_INVALID_VALUE_ID, &output_val_id,
        params->keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0));
  }

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
