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

#include "tensorflow/lite/experimental/lrt/core/dispatch/dispatch_delegate_kernel.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch_delegate.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/core/dispatch/dispatch_delegate_options.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"
#include "tensorflow/lite/experimental/lrt/core/tfl_utils.h"
#include "tensorflow/lite/experimental/lrt/core/utils.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_dispatch.h"

namespace lrt {
namespace internal {

DispatchDelegateKernel::~DispatchDelegateKernel() {
  for (size_t i = 0; i < input_tensor_buffer_handles_.size(); ++i) {
    (void)LrtDispatchDetachInput(invocation_context_, i,
                                 input_tensor_buffer_handles_[i]);
  }

  for (size_t i = 0; i < output_tensor_buffer_handles_.size(); ++i) {
    (void)LrtDispatchDetachOutput(invocation_context_, i,
                                  output_tensor_buffer_handles_[i]);
  }

  if (invocation_context_) {
    (void)LrtDispatchInvocationContextDestroy(invocation_context_);
  }

  for (auto& buffer_handle : input_tensor_buffer_handles_) {
    (void)LrtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
  }

  for (auto& buffer_handle : output_tensor_buffer_handles_) {
    (void)LrtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
  }

  if (device_context_) {
    (void)LrtDispatchDeviceContextDestroy(device_context_);
  }

  input_tensor_buffers_.clear();
  output_tensor_buffers_.clear();
}

absl::StatusOr<DispatchDelegateKernel::Ptr> DispatchDelegateKernel::Create(
    std::string&& graph_name, const LrtDispatchDelegateOptions& options) {
  auto dispatch_api_lib_path =
      options.GetOption(LrtDispatchDelegateOptions::kDispatchApiLibPath);
  if (auto status = LrtDispatchInitialize(dispatch_api_lib_path.has_value()
                                              ? dispatch_api_lib_path->data()
                                              : nullptr);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to initialize Dispatch API: %d", status);
    return absl::InternalError("Failed to initialize Dispatch API");
  }

  const char* vendor_id;
  if (auto status = LrtDispatchGetVendorId(&vendor_id);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get Dispatch API vendor ID: %d", status);
    return absl::InternalError("Failed to get Dispatch API vendor ID");
  }
  LITE_RT_LOG(LRT_INFO, "Dispatch API vendor ID: %s", vendor_id);

  const char* build_id;
  if (auto status = LrtDispatchGetBuildId(&build_id); status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get Dispatch API build ID: %d", status);
    return absl::InternalError("Failed to get Dispatch API build ID");
  }
  LITE_RT_LOG(LRT_INFO, "Dispatch API build ID: %s", build_id);

  LrtDispatchApiVersion api_version;
  if (auto status = LrtDispatchGetApiVersion(&api_version);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get Dispatch API version: %d", status);
    return absl::InternalError("Failed to get Dispatch API version");
  }
  LITE_RT_LOG(LRT_INFO, "Dispatch API version: %d.%d.%d", api_version.major,
              api_version.minor, api_version.patch);
  // Check if the versions mach.
  if (api_version.major != LRT_DISPATCH_API_VERSION_MAJOR ||
      api_version.minor < LRT_DISPATCH_API_VERSION_MINOR) {
    return absl::InternalError(
        "Found Dispatch API with an unsupported version");
  }

  int capabilities;
  if (auto status = LrtDispatchGetCapabilities(&capabilities);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get Dispatch API capabilities: %d",
                status);
    return absl::InternalError("Failed to get Dispatch API capabilities");
  }
  LITE_RT_LOG(LRT_INFO, "Dispatch API capabilities: %d", capabilities);

  if (!(capabilities & kLrtDispatchCapabilitiesBasic)) {
    return absl::InternalError("Dispatch API has insufficient capabilities");
  }

  LrtDispatchDeviceContext device_context;
  if (auto status = LrtDispatchDeviceContextCreate(&device_context);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get Dispatch API device context: %d",
                status);
    return absl::InternalError("Failed to create Dispatch API device context");
  }

  return Ptr(new DispatchDelegateKernel(options, std::move(graph_name),
                                        device_context));
}

TfLiteStatus DispatchDelegateKernel::Init(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {
  LITE_RT_LOG(LRT_INFO, "DispatchDelegateKernel::Init");
  if (params->nodes_to_replace->size != 1) {
    LITE_RT_LOG(
        LRT_ERROR,
        "Models with more than one dispatch node are not yet supported");
    return kTfLiteError;
  }

  auto node_id = params->nodes_to_replace->data[0];
  TfLiteOpaqueNode* node;
  TfLiteOperator* op;
  if (auto status = TfLiteOpaqueContextGetNodeAndRegistration(context, node_id,
                                                              &node, &op);
      status != kTfLiteOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get node and registration: %d", status);
    return status;
  }

  const void* init_data;
  int init_data_size;
  if (auto status = TfLiteOpaqueNodeGetCustomInitialData(node, &init_data,
                                                         &init_data_size);
      status != kTfLiteOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to get custom initial data: %d", status);
    return status;
  }
  if (!init_data || !init_data_size) {
    LITE_RT_LOG(LRT_ERROR, "Found custom op with missing initial data");
    return kTfLiteError;
  }

  std::string custom_option(static_cast<const char*>(init_data),
                            init_data_size);
  auto exec_info = options_.GetExecInfo(custom_option);
  if (!exec_info.ok()) {
    LITE_RT_LOG(LRT_ERROR, "Failed to fetch ExecInfo for %s: %s",
                custom_option.data(), exec_info.status().message().data());
    return kTfLiteError;
  }

  const char* function_name = exec_info->function_name.has_value()
                                  ? exec_info->function_name->data()
                                  : nullptr;
  int num_inputs = params->input_tensors->size;
  int num_outputs = params->output_tensors->size;
  if (auto status = LrtDispatchInvocationContextCreate(
          device_context_, kLrtDispatchExecutableTypeMlModel,
          exec_info->bytecode.data(), exec_info->bytecode.size(), function_name,
          num_inputs, num_outputs, &invocation_context_);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to create invocation context: %d", status);
    return kTfLiteError;
  }

  input_tensor_buffers_.resize(num_inputs);
  input_tensor_buffer_handles_.resize(num_inputs);
  input_tensor_buffer_used_size_.resize(num_inputs);

  output_tensor_buffers_.resize(num_outputs);
  output_tensor_buffer_handles_.resize(num_outputs);
  output_tensor_buffer_used_size_.resize(num_outputs);

  return kTfLiteOk;
}

absl::StatusOr<TensorBufferRequirements>
DispatchDelegateKernel::GetBufferRequirements(
    const RankedTensorType& tensor_type, int io_tensor_index,
    bool is_input) const {
  auto lrt_tensor_type = static_cast<LrtRankedTensorType>(tensor_type);
  LrtTensorBufferRequirements tensor_buffer_requirements;
  if (is_input) {
    if (auto status = LrtDispatchGetInputRequirements(
            invocation_context_, /*input_index=*/io_tensor_index,
            &lrt_tensor_type, &tensor_buffer_requirements);
        status != kLrtStatusOk) {
      LITE_RT_LOG(LRT_ERROR,
                  "Failed to get tensor buffer requirements for input %d: %d",
                  io_tensor_index, status);
      return absl::InternalError(
          "Failed to get tensor buffer requirements for input");
    }

  } else {
    if (auto status = LrtDispatchGetOutputRequirements(
            invocation_context_, /*output_index=*/io_tensor_index,
            &lrt_tensor_type, &tensor_buffer_requirements);
        status != kLrtStatusOk) {
      LITE_RT_LOG(LRT_ERROR,
                  "Failed to get tensor buffer requirements for output %d: %d",
                  io_tensor_index, status);
      return absl::InternalError(
          "Failed to get tensor buffer requirements for output");
    }
  }

  return TensorBufferRequirements(tensor_buffer_requirements, /*owned=*/false);
}

TfLiteStatus DispatchDelegateKernel::SetBuffer(
    const TfLiteOpaqueTensor* tfl_opaque_tensor, int buffer_index,
    bool is_input) {
  auto& cached_tensor_buffer = is_input ? input_tensor_buffers_[buffer_index]
                                        : output_tensor_buffers_[buffer_index];
  auto& cached_tensor_buffer_handle =
      is_input ? input_tensor_buffer_handles_[buffer_index]
               : output_tensor_buffer_handles_[buffer_index];
  auto& used_size = is_input ? input_tensor_buffer_used_size_[buffer_index]
                             : output_tensor_buffer_used_size_[buffer_index];

  auto tensor_type = ConvertTensorType(tfl_opaque_tensor);
  if (!tensor_type.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", tensor_type.status().message().data());
    return kTfLiteError;
  }

  // Check if we can reuse a cached tensor buffer or we need to create a new
  // one.
  if (cached_tensor_buffer.IsValid()) {
    if (auto cached_tensor_type = cached_tensor_buffer.TensorType();
        !cached_tensor_type.ok()) {
      LITE_RT_LOG(LRT_ERROR, "%s",
                  cached_tensor_type.status().message().data());
      return kTfLiteError;
    }

    if (tensor_type->Layout() == cached_tensor_buffer.TensorType()->Layout()) {
      // We can reuse the cached tensor buffer.
      return kTfLiteOk;
    }

    // We cannot reuse the cached tensor buffer; proceed below.
  }

  auto tensor_buffer_requirements =
      GetBufferRequirements(*tensor_type, buffer_index, is_input);
  if (!tensor_buffer_requirements.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s",
                tensor_buffer_requirements.status().message().data());
    return kTfLiteError;
  }

  auto supported_tensor_buffer_types =
      tensor_buffer_requirements->SupportedTypes();
  if (!supported_tensor_buffer_types.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s",
                supported_tensor_buffer_types.status().message().data());
    return kTfLiteError;
  }

  if (supported_tensor_buffer_types->empty()) {
    LITE_RT_LOG(LRT_ERROR,
                "Insufficient number of supported tensor buffer types");
    return kTfLiteError;
  }

  // For now we simply pick the first buffer type that's supported.
  LrtTensorBufferType tensor_buffer_type = (*supported_tensor_buffer_types)[0];

  auto tensor_buffer_size = tensor_buffer_requirements->BufferSize();
  if (!tensor_buffer_size.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", tensor_buffer_size.status().message().data());
    return kTfLiteError;
  }

  auto lrt_tensor_type = static_cast<LrtRankedTensorType>(*tensor_type);
  LrtTensorBuffer lrt_tensor_buffer;
  if (auto status =
          LrtCreateManagedTensorBuffer(tensor_buffer_type, &lrt_tensor_type,
                                       *tensor_buffer_size, &lrt_tensor_buffer);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to create managed tensor buffer: %d",
                status);
    return kTfLiteError;
  }

  TensorBuffer tensor_buffer(lrt_tensor_buffer);

  LrtTensorBufferHandle buffer_handle;
  if (auto status = LrtDispatchRegisterTensorBuffer(
          device_context_, static_cast<LrtTensorBuffer>(tensor_buffer),
          &buffer_handle);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to register tensor buffer: %d", status);
    return kTfLiteError;
  }

  if (is_input) {
    if (auto status = LrtDispatchAttachInput(invocation_context_, buffer_index,
                                             buffer_handle);
        status != kLrtStatusOk) {
      (void)LrtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
      LITE_RT_LOG(LRT_ERROR, "Failed to attach tensor buffer to input %d: %d",
                  buffer_index, status);
      return kTfLiteError;
    }
  } else {
    if (auto status = LrtDispatchAttachOutput(invocation_context_, buffer_index,
                                              buffer_handle);
        status != kLrtStatusOk) {
      (void)LrtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
      LITE_RT_LOG(LRT_ERROR, "Failed to attach tensor buffer to output %d: %d",
                  buffer_index, status);
      return kTfLiteError;
    }
  }

  auto num_bytes = internal::GetNumPackedBytes(
      static_cast<LrtRankedTensorType>(*tensor_type));
  if (!num_bytes.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", num_bytes.status().message().data());
    return kTfLiteError;
  }

  cached_tensor_buffer = std::move(tensor_buffer);
  cached_tensor_buffer_handle = buffer_handle;
  used_size = *num_bytes;

  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Prepare(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueNode* node) {
  size_t num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  for (size_t i = 0; i < num_node_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    if (auto status = SetBuffer(tfl_opaque_tensor, i, /*is_input=*/true);
        status != kTfLiteOk) {
      return status;
    }
  }

  size_t num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  for (size_t i = 0; i < num_node_outputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    if (auto status = SetBuffer(tfl_opaque_tensor, i, /*is_input=*/false);
        status != kTfLiteOk) {
      return status;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                          TfLiteOpaqueNode* node) {
  size_t num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  if (num_node_inputs != input_tensor_buffers_.size()) {
    LITE_RT_LOG(LRT_ERROR, "Invalid number of inputs");
    return kTfLiteError;
  }

  for (size_t i = 0; i < num_node_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    void* tensor_data = TfLiteOpaqueTensorData(tfl_opaque_tensor);
    auto& tensor_buffer = input_tensor_buffers_[i];

    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    if (!lock_and_addr.ok()) {
      LITE_RT_LOG(LRT_ERROR, "%s", lock_and_addr.status().message().data());
      return kTfLiteError;
    }

    size_t buffer_size = input_tensor_buffer_used_size_[i];
    std::memcpy(lock_and_addr->second, tensor_data, buffer_size);
  }

  if (auto status = LrtDispatchInvoke(invocation_context_);
      status != kLrtStatusOk) {
    LITE_RT_LOG(LRT_ERROR, "Failed to invoke context: %d", status);
    return kTfLiteError;
  }

  size_t num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  if (num_node_outputs != output_tensor_buffers_.size()) {
    LITE_RT_LOG(LRT_ERROR, "Invalid number of outputs");
    return kTfLiteError;
  }

  for (size_t i = 0; i < num_node_outputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    void* tensor_data = TfLiteOpaqueTensorData(tfl_opaque_tensor);
    auto& tensor_buffer = output_tensor_buffers_[i];

    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    if (!lock_and_addr.ok()) {
      LITE_RT_LOG(LRT_ERROR, "%s", lock_and_addr.status().message().data());
      return kTfLiteError;
    }

    size_t buffer_size = output_tensor_buffer_used_size_[i];
    std::memcpy(tensor_data, lock_and_addr->second, buffer_size);
  }

  return kTfLiteOk;
}

}  // namespace internal
}  // namespace lrt
