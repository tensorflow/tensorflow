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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace example {
namespace helpers {
int CalculateNumElements(const TfLiteOpaqueTensor* opaque_tensor);
}  // namespace helpers

// LINT.IfChange
static const char kSampleStableDelegateName[] = "google_sample_delegate";
// LINT.ThenChange(Google-internal path)
static const char kSampleStableDelegateVersion[] = "1.0.0";

// A simple delegate that supports only addition and subtraction operations.
// Implements SimpleOpaqueDelegateInterface, and therefore the delegate can be
// easily be adapted to work with the stable TFLite delegate API via
// TfLiteOpaqueDelegateFactory.
class SampleStableDelegate : public SimpleOpaqueDelegateInterface {
 public:
  // SampleStableDelegate supports float32 input type only.
  // Returns true if the inputs of 'node' are two tensors of float32 with the
  // same shape and the operation is addition or subtraction (without fused
  // activation).
  bool IsNodeSupportedByDelegate(
      const TfLiteRegistrationExternal* registration_external,
      const TfLiteOpaqueNode* node,
      TfLiteOpaqueContext* context) const override;

  // No-op. The delegate doesn't have extra steps to perform during
  // initialization.
  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  // Returns a name that identifies the delegate.
  const char* Name() const override;

  // Returns an instance of SampleStableDelegateKernel that implements
  // SimpleOpaqueDelegateKernelInterface. SampleStableDelegateKernel describes
  // how a subgraph is delegated and the concrete evaluation of both addition
  // and subtraction operations to be performed by the delegate.
  std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;
};

}  // namespace example
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_H_
