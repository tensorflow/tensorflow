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

#include "tensorflow/lite/delegates/flex/training/training_delegate.h"

namespace tflite {
namespace flex {

// Corresponding weak declaration found in lite/interpreter_builder.cc.
#if TFLITE_HAS_ATTRIBUTE_WEAK
// If weak symbol is not supported (Windows), it can use
// TF_AcquireFlexDelegate() path instead.
TfLiteDelegateUniquePtr AcquireFlexDelegate() { return {nullptr, nullptr}; }
#endif

TrainingFlexDelegate::TrainingFlexDelegate()
    : delegate_(FlexDelegate::Create()) {
  cancellation_manager_ = absl::make_unique<tensorflow::CancellationManager>();
  auto* flex_delegate = static_cast<FlexDelegate*>(delegate_->data_);
  auto* delegate_data = flex_delegate->mutable_data();
  delegate_data->SetCancellationManager(cancellation_manager_.get());
}

void TrainingFlexDelegate::Cancel() { cancellation_manager_->StartCancel(); }

bool TrainingFlexDelegate::ShouldCancel(void* data) {
  if (data == nullptr) {
    return false;
  }

  auto* training_flex_delegate = static_cast<TrainingFlexDelegate*>(data);
  return training_flex_delegate->cancellation_manager_->IsCancelled();
}

}  // namespace flex
}  // namespace tflite

// Exported C interface function which is used by AcquireFlexDelegate() at
// interpreter_builder.cc. To export the function name globally, the function
// name must be matched with patterns in tf_version_script.lds. In Android, we
// don't use this feature so skip building.
#if !defined(__ANDROID__)
extern "C" {
TFL_CAPI_EXPORT tflite::TfLiteDelegateUniquePtr TF_AcquireFlexDelegate() {
  return {nullptr, nullptr};
}
}  // extern "C"
#endif  // !defined(__ANDROID__)
