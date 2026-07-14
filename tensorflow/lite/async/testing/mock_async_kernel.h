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
#ifndef TENSORFLOW_LITE_ASYNC_TESTING_MOCK_ASYNC_KERNEL_H_
#define TENSORFLOW_LITE_ASYNC_TESTING_MOCK_ASYNC_KERNEL_H_

#include <vector>

#include <gmock/gmock.h>
#include "tensorflow/lite/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/async/c/types.h"

namespace tflite {
namespace async {
namespace testing {

// A fully mocked out async kernel.
// Mocked TfLiteAsyncKernel can be retreived by `MockAsyncKernel::kernel()`.
class MockAsyncKernel : public delegates::BackendAsyncKernelInterface {
 public:
  MOCK_METHOD(TfLiteStatus, RegisterBuffer,
              (TfLiteOpaqueContext*, TfLiteIoType, const TfLiteBackendBuffer*,
               const TfLiteAttributeMap*, TfLiteBufferHandle),
              (override));
  MOCK_METHOD(TfLiteStatus, RegisterBufferSlice,
              (TfLiteOpaqueContext*, TfLiteBufferHandle,
               const TfLiteAttributeMap*, TfLiteBufferHandle),
              (override));
  MOCK_METHOD(TfLiteStatus, UnregisterBuffer,
              (TfLiteOpaqueContext*, TfLiteBufferHandle), (override));
  MOCK_METHOD(bool, ReconcileRestrictions,
              (const TfLiteOpaqueContext*, const TfLiteOpaqueNode*, int,
               const TfLiteAttributeMap*, TfLiteAttributeMap*,
               TfLiteAttributeMap*),
              (const, override));
  MOCK_METHOD(TfLiteStatus, SetAttributes,
              (TfLiteOpaqueContext*, TfLiteOpaqueNode*, int,
               const TfLiteAttributeMap*),
              (override));
  MOCK_METHOD(TfLiteStatus, SetBufferAttributes,
              (const TfLiteBackendBuffer*, const TfLiteAttributeMap*),
              (override));
  MOCK_METHOD(TfLiteStatus, GetBufferAttributes,
              (const TfLiteBackendBuffer*, TfLiteAttributeMap*), (override));
  MOCK_METHOD(TfLiteStatus, Prepare, (TfLiteOpaqueContext*, TfLiteOpaqueNode*),
              (override));
  MOCK_METHOD(TfLiteStatus, Eval,
              (TfLiteOpaqueContext*, TfLiteOpaqueNode*, TfLiteExecutionTask*),
              (override));
  MOCK_METHOD(TfLiteStatus, Wait, (TfLiteOpaqueContext*, TfLiteExecutionTask*),
              (override));
  MOCK_METHOD(TfLiteStatus, Finish,
              (TfLiteOpaqueContext*, TfLiteExecutionTask*), (override));

  const std::vector<const char*>& SupportedBufferTypes(
      TfLiteIoType io_type) const override {
    return buffer_types_;
  }
  const std::vector<const char*>& SupportedSynchronizations(
      TfLiteIoType io_type) const override {
    return sync_types_;
  }

 private:
  const std::vector<const char*> buffer_types_{"buffer_type"};
  const std::vector<const char*> sync_types_{"sync_type"};
};

}  // namespace testing
}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_ASYNC_TESTING_MOCK_ASYNC_KERNEL_H_
