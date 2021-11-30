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

#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_TRAINING_TRAINING_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_TRAINING_TRAINING_DELEGATE_H_

#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/lite/delegates/flex/delegate.h"

namespace tflite {
namespace flex {

namespace testing {
class TrainingDelegateTest;
}  // namespace testing

// This is an experimental interface that is subject to change.
// Delegate that extends the functionality of FlexDelegate and supports
// cancellation.
//
// The interpreter must be constructed after and destructed
// before this delegate.
//
// Usage:
//   auto delegate = absl::make_unique<TrainingFlexDelegate>();
//   InterpreterBuilder builder;
//   builder.AddDelegate(delegate->GetTfLiteDelegate());
//   ... build interpreter ...
//
//   interpreter_->SetCancellationFunction(
//     delegate.get(),
//     TrainingFlexDelegate::ShouldCancel);
//
//   ... run inference ...
//
//   delegate->Cancel();
//
//   ... destroy interpreter ...
//   ... destroy delegate ...
class TrainingFlexDelegate {
 public:
  friend class testing::TrainingDelegateTest;

  TrainingFlexDelegate();

  // This method is thread safe. It does two things:
  //   1. Calls the CancellationManager of the TF eager runtime to support
  //      intra-op cancellation in TF.
  //   2. Uses the CancellationManager to signal TFLite interpreter for inter-op
  //      cancellation.
  // Training is non-recoverable after calling this API.
  void Cancel();

  TfLiteDelegate* GetTfLiteDelegate() const { return delegate_.get(); }

  // The param `data` must be a pointer to a TrainingFlexDelegate instance.
  static bool ShouldCancel(void* data);

 private:
  TfLiteDelegateUniquePtr delegate_;
  std::unique_ptr<tensorflow::CancellationManager> cancellation_manager_;
};

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_TRAINING_TRAINING_DELEGATE_H_
