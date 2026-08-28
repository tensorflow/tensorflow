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

#include "tensorflow/lite/delegates/ynnpack/pooling.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

TfLiteStatus IsPoolingSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node,
                                TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context, IsTensorSupported(input));
  TF_LITE_ENSURE(context, IsTensorSupported(output));

  TF_LITE_ENSURE_EQ(context, input.type, output.type);
  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  // We only support NHWC format for 2D pooling, which means 4D tensor. NHWC
  // format NHWC is represented as [batch, height, width, channels].
  TF_LITE_ENSURE_EQ(context, input.dims->size, 4);

  const auto* params = static_cast<const TfLitePoolParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);
  TF_LITE_ENSURE(context, params->filter_height > 0);
  TF_LITE_ENSURE(context, params->filter_width > 0);

  TF_LITE_ENSURE(context,
                 IsActivationSupported(params->activation, output.type));

  return kTfLiteOk;
}

TfLiteStatus DefineMaxPool2DNode(TfLiteContext* context,
                                 ynn_subgraph_t subgraph,
                                 TensorToValueIdMap& tensor_to_value_id,
                                 const NodeInfo& node) {
  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLitePoolParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];
  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  uint32_t input_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         input_tensor_index);
  uint32_t output_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                          output_tensor_index);

  TF_LITE_ENSURE(context, input_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_id != YNN_INVALID_VALUE_ID);

  TfLiteFusedActivation activation = node.activation;

  uint32_t maxpool_output_id =
      activation == kTfLiteActNone ? output_id : YNN_INVALID_VALUE_ID;

  // Pad with -inf for max pooling.
  float padding_val = -std::numeric_limits<float>::infinity();
  if (IsQuantized(input_tensor)) {
    if (input_tensor.type == kTfLiteInt8) {
      padding_val = std::numeric_limits<int8_t>::min();
    } else if (input_tensor.type == kTfLiteUInt8) {
      padding_val = std::numeric_limits<uint8_t>::min();
    }
  }

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineYnnStencil(
      context, subgraph, input_tensor, input_id, params->filter_height,
      params->filter_width, params->stride_height, params->stride_width,
      /*dilation_height=*/1, /*dilation_width=*/1, params->padding, padding_val,
      &stencil_id));

  const int32_t reduce_axes[] = {3, 4};
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_max, 2, reduce_axes, stencil_id,
                        YNN_INVALID_VALUE_ID, &maxpool_output_id, /*flags=*/0));

  if (activation != kTfLiteActNone) {
    TF_LITE_ENSURE_STATUS(ApplyActivation(
        context, subgraph, activation, maxpool_output_id, output_id,
        output_tensor_index, GetYnnType(output_tensor.type)));
  }

  tensor_to_value_id[output_tensor_index] = output_id;
  return kTfLiteOk;
}

TfLiteStatus DefineAveragePool2DNode(TfLiteContext* context,
                                     ynn_subgraph_t subgraph,
                                     TensorToValueIdMap& tensor_to_value_id,
                                     const NodeInfo& node) {
  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLitePoolParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  return DefineDecomposedUnaryNode(
      context, subgraph, tensor_to_value_id, node,
      [context, subgraph, params](uint32_t input_id,
                                  uint32_t& output_id) -> TfLiteStatus {
        bool padding_same = (params->padding == kTfLitePaddingSame);
        TF_LITE_ENSURE_YNN_STATUS(ynn::define_average_pool_2d(
            subgraph, input_id, ynn_type_fp32, padding_same,
            params->filter_height, params->filter_width, params->stride_height,
            params->stride_width, output_id));
        return kTfLiteOk;
      });
}

}  // namespace ynnpack
}  // namespace tflite
