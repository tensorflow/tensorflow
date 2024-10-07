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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch/registry.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn.h"

class LrtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LrtDispatchDeviceContextT>;

  ~LrtDispatchDeviceContextT();

  static absl::StatusOr<Ptr> Create(const lrt::qnn::Qnn& qnn);

  Qnn_BackendHandle_t backend_handle() { return backend_handle_; }

  absl::StatusOr<LrtTensorBufferHandle> RegisterTensorBuffer(
      LrtTensorBuffer tensor_buffer) {
    return tensor_buffer_registry_.Register(
        TensorBufferRegistryEntry(tensor_buffer));
  }

  absl::Status UnregisterTensorBuffer(
      LrtTensorBufferHandle tensor_buffer_handle) {
    return tensor_buffer_registry_.Unregister(tensor_buffer_handle);
  }

  absl::StatusOr<LrtTensorBuffer> GetTensorBuffer(
      LrtTensorBufferHandle tensor_buffer_handle);

  absl::StatusOr<Qnn_MemHandle_t> GetMemHandle(
      LrtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor,
      Qnn_ContextHandle_t context_handle);

 private:
  struct TensorBufferRegistryEntry {
    LrtTensorBuffer tensor_buffer;
    Qnn_MemHandle_t qnn_mem_handle = nullptr;
    explicit TensorBufferRegistryEntry(LrtTensorBuffer tensor_buffer_)
        : tensor_buffer(tensor_buffer_) {}
  };

  using TensorBufferRegistry =
      lrt::qnn::Registry<LrtTensorBufferHandle, TensorBufferRegistryEntry>;

  LrtDispatchDeviceContextT(const lrt::qnn::Qnn& qnn,
                            Qnn_BackendHandle_t backend_handle)
      : qnn_(qnn), backend_handle_(backend_handle) {}

  absl::StatusOr<Qnn_MemHandle_t> RegisterTensorBuffer(
      LrtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor,
      Qnn_ContextHandle_t context_handle);

  const lrt::qnn::Qnn& qnn_;
  Qnn_BackendHandle_t backend_handle_;
  TensorBufferRegistry tensor_buffer_registry_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_
