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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {

namespace flex {
namespace testing {
class KernelTest;
}  // namespace testing
}  // namespace flex

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
//     interpreter->ModifyGraphWithDelegate(std::move(delegate));
//   }
//   ... run inference ...
//   ... destroy interpreter ...
class FlexDelegate : public SimpleDelegateInterface {
 public:
  friend class flex::testing::KernelTest;

  // Creates a delegate that supports TF ops.
  static TfLiteDelegateUniquePtr Create() {
    return Create(/*base_delegate*/ nullptr);
  }

  ~FlexDelegate() override {}

  flex::DelegateData* mutable_data() { return &delegate_data_; }

 protected:
  // We sometimes have to create certain stub data to test FlexDelegate. To
  // achieve this, we will make a testing flex delegate class that inherits from
  // FlexDelegate to override certain things for stub data creation. Therefore,
  // this function accepts a FlexDelegate instance to initiliaze it properly for
  // create a testing flex delegate in some cases, and it is only used in
  // testing.
  static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<FlexDelegate> base_delegate);

  FlexDelegate() {}

  const char* Name() const override;

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override;

  TfLiteStatus Initialize(TfLiteContext* context) override;

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override;

  TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteTensor* output);

  flex::DelegateData delegate_data_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
