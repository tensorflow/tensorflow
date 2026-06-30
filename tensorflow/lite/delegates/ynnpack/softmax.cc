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

#include "tensorflow/lite/delegates/ynnpack/softmax.h"

#include <cstdint>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

TfLiteStatus IsSoftmaxSupported(const TfLiteRegistration* registration,
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

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  return kTfLiteOk;
}

TfLiteStatus DefineSoftmaxNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                               TensorToValueIdMap& tensor_to_value_id,
                               const NodeInfo& node) {
  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteSoftmaxParams*>(tflite_node->builtin_data);
  float beta = params ? params->beta : 1.0f;

  return DefineDecomposedUnaryNode(
      context, subgraph, tensor_to_value_id, node,
      [context, subgraph, beta](uint32_t input_id,
                                uint32_t& output_id) -> TfLiteStatus {
        TF_LITE_ENSURE_YNN_STATUS(
            ynn::define_softmax(subgraph, input_id, beta, output_id));
        return kTfLiteOk;
      });
}

TfLiteStatus DefineLogSoftmaxNode(TfLiteContext* context,
                                  ynn_subgraph_t subgraph,
                                  TensorToValueIdMap& tensor_to_value_id,
                                  const NodeInfo& node) {
  return DefineDecomposedUnaryNode(
      context, subgraph, tensor_to_value_id, node,
      [context, subgraph](uint32_t input_id,
                          uint32_t& output_id) -> TfLiteStatus {
        TF_LITE_ENSURE_YNN_STATUS(
            ynn::define_log_softmax(subgraph, input_id, output_id));
        return kTfLiteOk;
      });
}

}  // namespace ynnpack
}  // namespace tflite
