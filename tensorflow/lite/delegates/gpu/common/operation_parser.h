/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATION_PARSER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATION_PARSER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Parses TFLite operation and updates provided GraphFloat32.
class TFLiteOperationParser {
 public:
  virtual ~TFLiteOperationParser() = default;

  // Parses TFLite operation. This method allows expanding fused operations
  // into more than one node.
  virtual absl::Status Parse(const TfLiteNode* tflite_node,
                             const TfLiteRegistration* registration,
                             GraphFloat32* graph, ObjectReader* reader) = 0;

  // Verifies whether passed tflite node may be built by GPU delegate or not.
  virtual absl::Status IsSupported(const TfLiteContext* context,
                                   const TfLiteNode* tflite_node,
                                   const TfLiteRegistration* registration) = 0;

  // Returns the value IDs in the graph that correspond to the updated values of
  // the variable input tensor.
  virtual absl::flat_hash_map<int, ValueId>
  GetNewValueIdsForVariableInputNodes() {
    return {};
  }
};

absl::Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                        int max_version);
HW ToHW(int32_t h, int32_t w);
absl::Status ParsePoolingAttributes(const TfLitePoolParams* tf_options,
                                    const BHWC& input_shape,
                                    Pooling2DAttributes* attr);

template <typename AttrT>
void UpdatePadding(const TfLitePadding& padding, const BHWC& input_shape,
                   AttrT* attr) {
  if (padding == kTfLitePaddingSame) {
    attr->padding = CalculateSamePadding(input_shape, *attr);
  } else {
    attr->padding.prepended = HW(0, 0);
    attr->padding.appended = HW(0, 0);
  }
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATION_PARSER_H_
