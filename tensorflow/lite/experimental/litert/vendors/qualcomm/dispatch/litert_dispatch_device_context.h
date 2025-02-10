// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch/registry.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT() = default;

  static litert::Expected<Ptr> Create(litert::qnn::QnnManager& qnn_manager);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer) {
    return tensor_buffer_registry_.Register(
        TensorBufferRegistryEntry(tensor_buffer));
  }

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    return tensor_buffer_registry_.Unregister(tensor_buffer_handle);
  }

  litert::Expected<LiteRtTensorBuffer> GetTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<Qnn_MemHandle_t> GetMemHandle(
      LiteRtTensorBufferHandle tensor_buffer_handle,
      const Qnn_Tensor_t& tensor);

  void SetInvocationContext(
      LiteRtDispatchInvocationContextT* invocation_context) {
    invocation_context_ = invocation_context;
  }

 private:
  struct TensorBufferRegistryEntry {
    LiteRtTensorBuffer tensor_buffer;
    Qnn_MemHandle_t qnn_mem_handle = nullptr;
    explicit TensorBufferRegistryEntry(LiteRtTensorBuffer tensor_buffer_)
        : tensor_buffer(tensor_buffer_) {}
  };

  using TensorBufferRegistry = litert::qnn::Registry<LiteRtTensorBufferHandle,
                                                     TensorBufferRegistryEntry>;

  LiteRtDispatchDeviceContextT(litert::qnn::QnnManager& qnn_manager)
      : qnn_manager_(qnn_manager) {}

  litert::Expected<Qnn_MemHandle_t> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor);

  litert::qnn::QnnManager& qnn_manager_;
  TensorBufferRegistry tensor_buffer_registry_;
  LiteRtDispatchInvocationContextT* invocation_context_ = nullptr;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
