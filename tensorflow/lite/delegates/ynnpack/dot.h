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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

bool IsRuntimeBmm(const TfLiteRegistration* registration,
                  const TfLiteNode* node);
bool IsRuntimeBmm(TfLiteContext* context, int node_index);

TfLiteStatus IsBatchMatMulSupported(const TfLiteRegistration* registration,
                                    const TfLiteNode* node,
                                    TfLiteContext* context,
                                    bool is_runtime_bmm = false);

TfLiteStatus IsRuntimeBatchedMatMulSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context);

TfLiteStatus IsFullyConnectedSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);

TfLiteStatus DefineBatchMatMulNode(TfLiteContext* context,
                                   ynn_subgraph_t subgraph,
                                   TensorToValueIdMap& tensor_to_value_id,
                                   const NodeInfo& node);

TfLiteStatus DefineRuntimeBatchedMatMulNode(
    TfLiteContext* context, ynn_subgraph_t subgraph,
    TensorToValueIdMap& tensor_to_value_id, uint32_t& next_external_id,
    std::vector<DummyInputInfo>& dummy_inputs, const NodeInfo& node);

TfLiteStatus DefineFullyConnectedNode(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      TensorToValueIdMap& tensor_to_value_id,
                                      const NodeInfo& node);

TfLiteStatus IsConvSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context);

TfLiteStatus IsDepthwiseConvSupported(const TfLiteRegistration* registration,
                                      const TfLiteNode* node,
                                      TfLiteContext* context);

TfLiteStatus DefineConvNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            const NodeInfo& node);

TfLiteStatus DefineDepthwiseConvNode(TfLiteContext* context,
                                     ynn_subgraph_t subgraph,
                                     TensorToValueIdMap& tensor_to_value_id,
                                     const NodeInfo& node);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_
