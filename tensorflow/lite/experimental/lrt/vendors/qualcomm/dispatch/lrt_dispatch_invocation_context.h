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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/context_binary_info.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch/registry.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

class LrtDispatchDeviceContextT;

class LrtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LrtDispatchInvocationContextT>;

  ~LrtDispatchInvocationContextT() = default;

  static absl::StatusOr<Ptr> Create(lrt::qnn::QnnManager& qnn_manager,
                                    LrtDispatchDeviceContextT& device_context,
                                    const void* exec_bytecode_ptr,
                                    size_t exec_bytecode_size,
                                    const char* function_name);

  absl::StatusOr<LrtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LrtRankedTensorType& tensor_type);
  absl::StatusOr<LrtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LrtRankedTensorType& tensor_type);

  absl::Status AttachInput(int graph_input_index,
                           LrtTensorBufferHandle tensor_buffer_handle);

  absl::Status AttachOutput(int graph_output_index,
                            LrtTensorBufferHandle tensor_buffer_handle);

  absl::Status Execute();

 private:
  LrtDispatchInvocationContextT(
      lrt::qnn::QnnManager& qnn_manager,
      const lrt::qnn::ContextBinaryInfo& context_binary_info,
      LrtDispatchDeviceContextT& device_context,
      Qnn_ProfileHandle_t profile_handle, int graph_index,
      Qnn_GraphHandle_t graph_handle);

  absl::Status AttachBuffer(Qnn_Tensor_t& tensor,
                            LrtTensorBufferHandle tensor_buffer_handle);

  lrt::qnn::QnnManager& qnn_manager_;
  LrtDispatchDeviceContextT& device_context_;
  Qnn_ProfileHandle_t profile_handle_;
  int graph_index_;
  Qnn_GraphHandle_t graph_handle_;
  std::vector<lrt::qnn::QnnTensor> inputs_;
  std::vector<lrt::qnn::QnnTensor> outputs_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_
