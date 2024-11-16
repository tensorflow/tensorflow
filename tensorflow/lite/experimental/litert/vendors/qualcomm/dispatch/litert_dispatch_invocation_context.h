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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/context_binary_info.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch/registry.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

class LiteRtDispatchDeviceContextT;

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      litert::qnn::QnnManager& qnn_manager,
      LiteRtDispatchDeviceContextT& device_context,
      const void* exec_bytecode_ptr, size_t exec_bytecode_size,
      const char* function_name);

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> Execute();

  Qnn_ContextHandle_t ContextHandle() { return context_handle_.get(); }

 private:
  LiteRtDispatchInvocationContextT(
      litert::qnn::QnnManager& qnn_manager,
      const litert::qnn::ContextBinaryInfo& context_binary_info,
      LiteRtDispatchDeviceContextT& device_context,
      litert::qnn::QnnManager::ContextHandle&& context_handle,
      Qnn_ProfileHandle_t profile_handle, int graph_index,
      Qnn_GraphHandle_t graph_handle);

  litert::Expected<void> AttachBuffer(
      Qnn_Tensor_t& tensor, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::qnn::QnnManager& qnn_manager_;
  LiteRtDispatchDeviceContextT& device_context_;
  litert::qnn::QnnManager::ContextHandle context_handle_;
  Qnn_ProfileHandle_t profile_handle_;
  int graph_index_;
  Qnn_GraphHandle_t graph_handle_;
  std::vector<litert::qnn::QnnTensor> inputs_;
  std::vector<litert::qnn::QnnTensor> outputs_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
