/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"

namespace tflite {
namespace flex {

// Data kept by the Flex delegate for the lifetime of an Interpreter.
//
// Note: This class is *not* thread-safe; any dependent delegates should not be
// used concurrently.
class DelegateData {
 public:
  DelegateData();
  ~DelegateData();

  // Prepare the necessary EagerContext and data for execution.
  // This must be called at least once before execution. After preparation
  // succeeds, redundant calls will be ignored (even if the session_options
  // differ).
  tensorflow::Status Prepare(const tensorflow::SessionOptions& session_options);

  // The EagerContext that is required for execution of Flex Ops.
  // Note: The context is lazily created after the first call to |Prepare()|.
  tensorflow::EagerContext* GetEagerContext() { return eager_context_; }

  // Map from TF Lite tensor index to TensorFlow tensor for a given context.
  BufferMap* GetBufferMap(const TfLiteContext* context) {
    return &buffer_map_[context];
  }

 private:
  // Will be null until Prepare() is called and completes successfully.
  tensorflow::EagerContext* eager_context_ = nullptr;
  // TODO(b/112439500): Clean up stale BufferMap instances after adding the
  // necessary cleanup hook from a TfLiteContext to a TfLiteDelegate.
  std::unordered_map<const TfLiteContext*, BufferMap> buffer_map_;
};

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
