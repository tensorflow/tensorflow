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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_H_

#include <string>

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

// TFliteDelegate to interface with ArmNN.
class ArmNNDelegate : public TfLiteDelegate {
 public:
  // Encapsulates all options that are specific to ArmNN delegate.
  struct Options {
    // Selected ArmNN backend with nul-terminated name.
    // Default to nullptr, which implies that ArmNN will pick the one it
    // considers best
    // from the available plugins
    // It is the caller's responsibility to ensure the string is valid for the
    // duration of the Options object lifetime.
    const char* backend_name = nullptr;
    // Enable tuning in case of a tunable backend
    bool enable_tuning = false;
    // Enable profiling information
    bool enable_profiling = false;
    // Enable profiling information (Log level : DEBUG)
    bool enable_logging = false;
  };

  // Uses default options.
  ArmNNDelegate();

  // The constructor that accepts options from user.
  explicit ArmNNDelegate(Options options);

  // Destructor
  ~ArmNNDelegate() = default;

  // Returns the delegate options.
  static const Options GetOptions(TfLiteDelegate* delegate);

 private:
  // Encapsulates all delegate data.
  struct Data {
    std::string backend_name;
    bool enable_tuning;
    bool enable_profiling;
    bool enable_logging;
  };

  // Implements TfLiteDelegate::Prepare.
  static TfLiteStatus DoPrepare(TfLiteContext* context,
                                TfLiteDelegate* delegate);

  // Copy the data from delegate buffer handle into raw memory of the given
  // 'tensor'. The delegate is allowed to allocate the raw
  // bytes as long as it follows the rules for kTfLiteDynamic tensors.
  static TfLiteStatus DoCopyFromBufferHandle(TfLiteContext* context,
                                             TfLiteDelegate* delegate,
                                             TfLiteBufferHandle buffer_handle,
                                             TfLiteTensor* tensor);

  // Copy the data from raw memory of the given 'tensor' to delegate buffer
  // handle. Currently this function is not supported, and calling the function
  // will result in an error.
  static TfLiteStatus DoCopyToBufferHandle(TfLiteContext* context,
                                           TfLiteDelegate* delegate,
                                           TfLiteBufferHandle buffer_handle,
                                           TfLiteTensor* tensor);

  // Free the Delegate Buffer Handle. Note: This only frees the handle, but
  // this doesn't release the underlying resource (e.g. textures). The
  // resources are either owned by application layer or the delegate.
  static void DoFreeBufferHandle(TfLiteContext* context,
                                 TfLiteDelegate* delegate,
                                 TfLiteBufferHandle* handle);

  Data delegate_data_;
};
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_H_
