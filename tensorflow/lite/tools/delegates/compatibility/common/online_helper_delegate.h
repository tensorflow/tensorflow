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

#ifndef TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_ONLINE_HELPER_DELEGATE_H_
#define TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_ONLINE_HELPER_DELEGATE_H_

#include <functional>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"

namespace tflite {
namespace tools {

// This utility delegate is required because some TfLiteContext related
// functions are forbidden if not calling in delegate. This class is only used
// for online mode.
// WARNING: This is an experimental class and subject to change.
class OnlineHelperDelegate : public TfLiteDelegate {
 public:
  OnlineHelperDelegate(
      std::unordered_map<std::string, std::string>& dcc_configs,
      std::function<absl::Status(TfLiteContext*, const TfLiteNode*,
                                 const TfLiteRegistration*,
                                 std::unordered_map<std::string, std::string>&,
                                 proto::OpCompatibilityResult*)>
          check_op_func_ptr,
      proto::CompatibilityResult* result)
      : TfLiteDelegate(TfLiteDelegateCreate()),
        result_(result),
        dcc_configs_(dcc_configs),
        check_op_func_ptr_(check_op_func_ptr) {
    Prepare = DoPrepare;
    CopyFromBufferHandle = DoCopyFromBufferHandle;
    CopyToBufferHandle = DoCopyToBufferHandle;
    FreeBufferHandle = DoFreeBufferHandle;
    data_ = &delegate_data_;
  }

 protected:
  // This function uses a pointer to a mehtod (implemented by each specific DCC)
  // which contains the logic to check whether the primary subgraph can be
  // delegated to the specific delegate.
  static TfLiteStatus DoPrepare(TfLiteContext* context,
                                TfLiteDelegate* delegate);

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
  proto::CompatibilityResult* result_;
  std::unordered_map<std::string, std::string> dcc_configs_;
  std::function<absl::Status(TfLiteContext*, const TfLiteNode*,
                             const TfLiteRegistration*,
                             std::unordered_map<std::string, std::string>&,
                             proto::OpCompatibilityResult*)>
      check_op_func_ptr_;
};

}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_ONLINE_HELPER_DELEGATE_H_
