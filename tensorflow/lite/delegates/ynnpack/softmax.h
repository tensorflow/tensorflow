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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_SOFTMAX_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_SOFTMAX_H_

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

TfLiteStatus IsSoftmaxSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node, TfLiteContext* context);

TfLiteStatus DefineSoftmaxNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                               TensorToValueIdMap& tensor_to_value_id,
                               const NodeInfo& node);

TfLiteStatus DefineLogSoftmaxNode(TfLiteContext* context,
                                  ynn_subgraph_t subgraph,
                                  TensorToValueIdMap& tensor_to_value_id,
                                  const NodeInfo& node);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_SOFTMAX_H_
