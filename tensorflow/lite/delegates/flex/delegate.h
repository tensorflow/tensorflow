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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"

namespace tflite {

// WARNING: This is an experimental interface that is subject to change.
// Delegate that can be used to extract parts of a graph that are designed to be
// executed by TensorFlow's runtime via Eager.
//
// The interpreter must be constructed after the FlexDelegate and destructed
// before the FlexDelegate. This delegate may be used with multiple
// interpreters, but it is *not* thread-safe.
//
// Usage:
//   auto delegate = FlexDelegate::Create();
//   ... build interpreter ...
//
//   if (delegate) {
//     interpreter->ModifyGraphWithDelegate(
//         delegate.get(), /*allow_dynamic_tensors=*/true);
//   }
//   ... run inference ...
//   ... destroy interpreter ...
//   ... destroy delegate ...
class FlexDelegate : public TfLiteDelegate {
 public:
  // Creates a delegate that supports TF ops.
  //
  // If the underyling TF Flex context creation fails, returns null.
  static std::unique_ptr<FlexDelegate> Create();

  ~FlexDelegate();

 private:
  FlexDelegate();

  flex::DelegateData delegate_data_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
