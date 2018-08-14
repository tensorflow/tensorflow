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
#ifndef TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_DELEGATE_H_
#define TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_DELEGATE_H_

#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/delegates/eager/delegate_data.h"
#include "tensorflow/contrib/lite/interpreter.h"

namespace tflite {

// WARNING: This is an experimental interface that is subject to change.
// Delegate that can be used to extract parts of a graph that are designed to be
// executed by TensorFlow's runtime via Eager.
//
// The interpreter must be constructed after the EagerDelegate and destructed
// before the EagerDelegate. This delegate can only be used with one
// interpreter.
//
// Usage:
//   EagerDelegate delegate;
//   ... build interpreter ...
//
//   delegate.Apply(interpreter);
//   ... run inference ...
//   ... destroy interpreter ...
//   ... destroy delegate ...
class EagerDelegate {
 public:
  EagerDelegate();
  ~EagerDelegate();

  // Modifies the graph loaded in the interpreter.
  TfLiteStatus Apply(Interpreter* interpreter);

 private:
  std::unique_ptr<eager::DelegateData> delegate_data_;
  std::unique_ptr<TfLiteDelegate> delegate_;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_DELEGATE_H_
