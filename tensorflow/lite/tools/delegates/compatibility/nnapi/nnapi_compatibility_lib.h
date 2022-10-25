/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_COMPATIBILITY_LIB_H_
#define TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_COMPATIBILITY_LIB_H_

#include <map>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"

namespace tflite {
namespace tools {

// Check if the given TFLite flatbuffer model is compatible with NNAPI delegate.
// WARNING: This is an experimental API and subject to change.
TfLiteStatus CheckCompatibility(
    TfLiteContext* context, int32_t runtime_feature_level,
    std::vector<int>* supported_nodes,
    std::map<int, std::vector<tflite::delegate::nnapi::NNAPIValidationFailure>>*
        failures_by_node);

// This utility delegate is required because some TfLiteContext related
// functions are forbidden if not calling in delegate.
// WARNING: This is an experimental class and subject to change.
class CompatibilityCheckerDelegate : public TfLiteDelegate {
 public:
  explicit CompatibilityCheckerDelegate(int32_t runtime_feature_level)
      : TfLiteDelegate(TfLiteDelegateCreate()),
        runtime_feature_level_(runtime_feature_level),
        supported_nodes_(),
        failures_by_node_() {
    Prepare = DoPrepare;
    CopyFromBufferHandle = DoCopyFromBufferHandle;
    CopyToBufferHandle = DoCopyToBufferHandle;
    FreeBufferHandle = DoFreeBufferHandle;
    data_ = &delegate_data_;
  }

  std::vector<int> GetSupportedNodes() { return supported_nodes_; }
  std::map<int, std::vector<tflite::delegate::nnapi::NNAPIValidationFailure>>
  GetFailuresByNode() {
    return failures_by_node_;
  }

 protected:
  static TfLiteStatus DoPrepare(TfLiteContext* context,
                                TfLiteDelegate* delegate) {
    auto self = reinterpret_cast<CompatibilityCheckerDelegate*>(delegate);
    TF_LITE_ENSURE_OK(context,
                      CheckCompatibility(context, self->runtime_feature_level_,
                                         &(self->supported_nodes_),
                                         &(self->failures_by_node_)));
    return kTfLiteOk;
  }

  // This function is not expected to be called in this delegate.
  static TfLiteStatus DoCopyFromBufferHandle(TfLiteContext* context,
                                             TfLiteDelegate* delegate,
                                             TfLiteBufferHandle buffer_handle,
                                             TfLiteTensor* tensor) {
    return kTfLiteError;
  }

  // This function is not expected to be called in this delegate.
  static TfLiteStatus DoCopyToBufferHandle(TfLiteContext* context,
                                           TfLiteDelegate* delegate,
                                           TfLiteBufferHandle buffer_handle,
                                           TfLiteTensor* tensor) {
    return kTfLiteError;
  }

  // There is no buffer handle in this delegate.
  static void DoFreeBufferHandle(TfLiteContext* context,
                                 TfLiteDelegate* delegate,
                                 TfLiteBufferHandle* handle) {}

 private:
  int delegate_data_;
  int runtime_feature_level_;
  std::vector<int> supported_nodes_;
  std::map<int, std::vector<tflite::delegate::nnapi::NNAPIValidationFailure>>
      failures_by_node_;
};

}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_COMPATIBILITY_LIB_H_
