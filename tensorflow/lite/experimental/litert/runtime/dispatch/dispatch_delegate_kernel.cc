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

#include "tensorflow/lite/experimental/litert/runtime/dispatch/dispatch_delegate_kernel.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/runtime/dispatch/dispatch_delegate_options.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/runtime/tfl_utils.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

namespace litert {
namespace internal {

namespace {

// Get the bytecode and function name from given custom op options data.
Expected<std::pair<absl::string_view, BufferRef<uint8_t>>> ResolveExecInfo(
    BufferRef<uint8_t> custom_opts, TfLiteOpaqueContext* context,
    const LiteRtDispatchDelegateOptions& options) {
  auto exec_info = ParseExecInfo(custom_opts);
  if (!exec_info) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse custom initial data", "");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  auto [function_name, metadata_key] = *exec_info;

  const char* metadata;
  size_t bytes;
  if (auto stat = TfLiteOpaqueContextGetMetadata(context, metadata_key.data(),
                                                 &metadata, &bytes);
      stat != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get metadata for dispatch op: %d",
               stat);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  BufferRef<uint8_t> metadata_buf(metadata, bytes);

  auto bytecode_loc = ParseByteCodePlaceholder(metadata_buf);
  if (!bytecode_loc) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse metadata for dispatch op", "");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  auto [bytecode_offset, bytecode_size] = *bytecode_loc;

  LITERT_LOG(
      LITERT_INFO,
      "Initializing invocation context for dispatch op\n\tfunction_name: "
      "%s\n\tbyte_code_offset: %lu \n\tbyte_code_size: %lu",
      function_name.data(), bytecode_offset, bytecode_size);
  if (bytecode_size == 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found zero-size bytecode");
  }

  auto alloc_base = FindAllocBase(options);
  if (!alloc_base) {
    LITERT_LOG(LITERT_ERROR,
               "Could not find requried delegate options \"alloc_base\"", "");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  const void* alloc = std::any_cast<const void*>(*alloc_base);
  const void* bytecode =
      reinterpret_cast<const uint8_t*>(alloc) + bytecode_offset;
  return std::make_pair(function_name,
                        BufferRef<uint8_t>(bytecode, bytecode_size));
}
}  // namespace

DispatchDelegateKernel::~DispatchDelegateKernel() {
  for (size_t i = 0; i < input_tensor_buffer_handles_.size(); ++i) {
    (void)LiteRtDispatchDetachInput(invocation_context_, i,
                                    input_tensor_buffer_handles_[i]);
  }

  for (size_t i = 0; i < output_tensor_buffer_handles_.size(); ++i) {
    (void)LiteRtDispatchDetachOutput(invocation_context_, i,
                                     output_tensor_buffer_handles_[i]);
  }

  if (invocation_context_) {
    (void)LiteRtDispatchInvocationContextDestroy(invocation_context_);
  }

  for (auto& buffer_handle : input_tensor_buffer_handles_) {
    (void)LiteRtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
  }

  for (auto& buffer_handle : output_tensor_buffer_handles_) {
    (void)LiteRtDispatchUnregisterTensorBuffer(device_context_, buffer_handle);
  }

  if (device_context_) {
    (void)LiteRtDispatchDeviceContextDestroy(device_context_);
  }

  input_tensor_buffers_.clear();
  output_tensor_buffers_.clear();
}

Expected<DispatchDelegateKernel::Ptr> DispatchDelegateKernel::Create(
    std::string&& graph_name, const LiteRtDispatchDelegateOptions& options) {
  auto dispatch_options = options.GetDispatchOptions();
  if (auto status = LiteRtDispatchInitialize(dispatch_options.data(),
                                             dispatch_options.size());
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to initialize Dispatch API: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to initialize Dispatch API");
  }

  const char* vendor_id;
  if (auto status = LiteRtDispatchGetVendorId(&vendor_id);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get Dispatch API vendor ID: %d",
               status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get Dispatch API vendor ID");
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API vendor ID: %s", vendor_id);

  const char* build_id;
  if (auto status = LiteRtDispatchGetBuildId(&build_id);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get Dispatch API build ID: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get Dispatch API build ID");
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API build ID: %s", build_id);

  LiteRtApiVersion api_version;
  if (auto status = LiteRtDispatchGetApiVersion(&api_version);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get LiteRT Dispatch API version: %d",
               status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get LiteRT Dispatch API version");
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API version: %d.%d.%d", api_version.major,
             api_version.minor, api_version.patch);
  // Check if the versions mach.
  if (api_version.major != LITERT_API_VERSION_MAJOR ||
      api_version.minor < LITERT_API_VERSION_MINOR) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found Dispatch API with an unsupported version");
  }

  int capabilities;
  if (auto status = LiteRtDispatchGetCapabilities(&capabilities);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get Dispatch API capabilities: %d",
               status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get Dispatch API capabilities");
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API capabilities: %d", capabilities);

  if (!(capabilities & kLiteRtDispatchCapabilitiesBasic)) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Dispatch API has insufficient capabilities");
  }

  LiteRtDispatchDeviceContext device_context;
  if (auto status = LiteRtDispatchDeviceContextCreate(&device_context);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get Dispatch API device context: %d",
               status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create Dispatch API device context");
  }

  return Ptr(new DispatchDelegateKernel(options, std::move(graph_name),
                                        device_context));
}

TfLiteStatus DispatchDelegateKernel::Init(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {
  if (params->nodes_to_replace->size != 1) {
    LITERT_LOG(LITERT_ERROR,
               "Models with more than one dispatch node are not yet supported");
    return kTfLiteError;
  }

  auto node_id = params->nodes_to_replace->data[0];
  TfLiteOpaqueNode* node;
  TfLiteOperator* op;
  if (auto status = TfLiteOpaqueContextGetNodeAndRegistration(context, node_id,
                                                              &node, &op);
      status != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get node and registration: %d", status);
    return status;
  }

  const void* init_data;
  int init_data_size;
  if (auto status = TfLiteOpaqueNodeGetCustomInitialData(node, &init_data,
                                                         &init_data_size);
      status != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get custom initial data: %d", status);
    return status;
  }
  if (!init_data || !init_data_size) {
    LITERT_LOG(LITERT_ERROR, "Found custom op with missing initial data");
    return kTfLiteError;
  }

  BufferRef<uint8_t> custom_opts(init_data, init_data_size);
  auto exec_info = ResolveExecInfo(custom_opts, context, options_);
  if (!exec_info) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse custom options");
    return kTfLiteError;
  }
  auto [function_name, bytecode] = *exec_info;

  const int num_inputs = params->input_tensors->size;
  const int num_outputs = params->output_tensors->size;

  if (auto status = LiteRtDispatchInvocationContextCreate(
          device_context_, kLiteRtDispatchExecutableTypeMlModel,
          bytecode.Data(), bytecode.Size(), function_name.data(), num_inputs,
          num_outputs, &invocation_context_);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: %d", status);
    return kTfLiteError;
  }

  input_tensor_buffers_require_cpu_sync_.resize(num_inputs);
  input_tensor_buffers_.resize(num_inputs);
  input_tensor_buffer_handles_.resize(num_inputs);
  input_tensor_buffer_used_size_.resize(num_inputs);

  output_tensor_buffers_require_cpu_sync_.resize(num_outputs);
  output_tensor_buffers_.resize(num_outputs);
  output_tensor_buffer_handles_.resize(num_outputs);
  output_tensor_buffer_used_size_.resize(num_outputs);

  void* external_context;
  TfLiteOpaqueContextGetExternalContext(context, &external_context,
                                        kTfLiteLiteRtBufferContext);
  if (!external_context) {
    LITERT_LOG(LITERT_ERROR, "External context not found");
    return kTfLiteError;
  }

  auto* buffer_context =
      reinterpret_cast<litert::internal::ExternalLiteRtBufferContext*>(
          external_context);

  // Register input and output buffer requirements.
  size_t num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  for (size_t i = 0; i < num_node_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    if (!tfl_opaque_tensor) {
      LITERT_LOG(LITERT_ERROR, "Failed to get TFL node input %d", i);
      return kTfLiteError;
    }
    auto tensor_type = ConvertTensorType(tfl_opaque_tensor);
    if (!tensor_type) {
      LITERT_LOG(LITERT_ERROR, "%s", tensor_type.Error().Message().data());
      return kTfLiteError;
    }
    auto input_buffer_requirements =
        GetBufferRequirements(*tensor_type, i, /*is_input=*/true);
    if (auto res = buffer_context->RegisterBufferRequirement(
            tfl_opaque_tensor, std::move(*input_buffer_requirements));
        res != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to register buffer requirement");
      return kTfLiteError;
    }
  }

  size_t num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  for (size_t i = 0; i < num_node_outputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    if (!tfl_opaque_tensor) {
      LITERT_LOG(LITERT_ERROR, "Failed to get TFL node output %d", i);
      return kTfLiteError;
    }
    auto tensor_type = ConvertTensorType(tfl_opaque_tensor);
    if (!tensor_type) {
      LITERT_LOG(LITERT_ERROR, "%s", tensor_type.Error().Message().data());
      return kTfLiteError;
    }
    auto output_buffer_requirements =
        GetBufferRequirements(*tensor_type, i, /*is_input=*/false);
    if (auto res = buffer_context->RegisterBufferRequirement(
            tfl_opaque_tensor, std::move(*output_buffer_requirements));
        res != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to register buffer requirement");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

Expected<TensorBufferRequirements>
DispatchDelegateKernel::GetBufferRequirements(
    const RankedTensorType& tensor_type, int io_tensor_index,
    bool is_input) const {
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LiteRtTensorBufferRequirements tensor_buffer_requirements;
  if (is_input) {
    if (auto status = LiteRtDispatchGetInputRequirements(
            invocation_context_, /*input_index=*/io_tensor_index,
            &litert_tensor_type, &tensor_buffer_requirements);
        status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to get tensor buffer requirements for input %d: %d",
                 io_tensor_index, status);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to get tensor buffer requirements for input");
    }

  } else {
    if (auto status = LiteRtDispatchGetOutputRequirements(
            invocation_context_, /*output_index=*/io_tensor_index,
            &litert_tensor_type, &tensor_buffer_requirements);
        status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to get tensor buffer requirements for output %d: %d",
                 io_tensor_index, status);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to get tensor buffer requirements for output");
    }
  }

  return TensorBufferRequirements(tensor_buffer_requirements,
                                  /*owned=*/true);
}

TfLiteStatus DispatchDelegateKernel::CreateAndSetBuffer(
    const TfLiteOpaqueTensor* tfl_opaque_tensor, int buffer_index,
    bool is_input) {
  auto& cached_tensor_buffer = is_input ? input_tensor_buffers_[buffer_index]
                                        : output_tensor_buffers_[buffer_index];

  auto tensor_type = ConvertTensorType(tfl_opaque_tensor);
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "%s", tensor_type.Error().Message().data());
    return kTfLiteError;
  }

  // Check if we can reuse a cached tensor buffer or we need to create a new
  // one.
  if (static_cast<bool>(cached_tensor_buffer)) {
    if (auto cached_tensor_type = cached_tensor_buffer.TensorType();
        !cached_tensor_type) {
      LITERT_LOG(LITERT_ERROR, "%s",
                 cached_tensor_type.Error().Message().data());
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
  if (!tensor_buffer_requirements) {
    LITERT_LOG(LITERT_ERROR, "%s",
               tensor_buffer_requirements.Error().Message().data());
    return kTfLiteError;
  }

  auto supported_tensor_buffer_types =
      tensor_buffer_requirements->SupportedTypes();
  if (!supported_tensor_buffer_types) {
    LITERT_LOG(LITERT_ERROR, "%s",
               supported_tensor_buffer_types.Error().Message().data());
    return kTfLiteError;
  }

  if (supported_tensor_buffer_types->empty()) {
    LITERT_LOG(LITERT_ERROR,
               "Insufficient number of supported tensor buffer types");
    return kTfLiteError;
  }

  // For now we simply pick the first buffer type that's supported.
  LiteRtTensorBufferType tensor_buffer_type =
      (*supported_tensor_buffer_types)[0];

  auto tensor_buffer_size = tensor_buffer_requirements->BufferSize();
  if (!tensor_buffer_size) {
    LITERT_LOG(LITERT_ERROR, "%s", tensor_buffer_size.Error().Message().data());
    return kTfLiteError;
  }

  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(*tensor_type);
  LiteRtTensorBuffer litert_tensor_buffer;
  if (auto status = LiteRtCreateManagedTensorBuffer(
          tensor_buffer_type, &litert_tensor_type, *tensor_buffer_size,
          &litert_tensor_buffer);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create managed tensor buffer: %d",
               status);
    return kTfLiteError;
  }

  return RegisterLiteRtTensorBuffer(TensorBuffer(litert_tensor_buffer),
                                    *tensor_buffer_size, buffer_index,
                                    is_input);
}

TfLiteStatus DispatchDelegateKernel::RegisterLiteRtTensorBuffer(
    TensorBuffer&& tensor_buffer, size_t buffer_used_size, int buffer_index,
    bool is_input) {
  LiteRtTensorBufferHandle buffer_handle;
  if (auto status = LiteRtDispatchRegisterTensorBuffer(
          device_context_, tensor_buffer.Get(), &buffer_handle);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to register tensor buffer: %d", status);
    return kTfLiteError;
  }

  if (is_input) {
    if (auto status = LiteRtDispatchAttachInput(invocation_context_,
                                                buffer_index, buffer_handle);
        status != kLiteRtStatusOk) {
      (void)LiteRtDispatchUnregisterTensorBuffer(device_context_,
                                                 buffer_handle);
      LITERT_LOG(LITERT_ERROR, "Failed to attach tensor buffer to input %d: %d",
                 buffer_index, status);
      return kTfLiteError;
    }
  } else {
    if (auto status = LiteRtDispatchAttachOutput(invocation_context_,
                                                 buffer_index, buffer_handle);
        status != kLiteRtStatusOk) {
      (void)LiteRtDispatchUnregisterTensorBuffer(device_context_,
                                                 buffer_handle);
      LITERT_LOG(LITERT_ERROR,
                 "Failed to attach tensor buffer to output %d: %d",
                 buffer_index, status);
      return kTfLiteError;
    }
  }

  if (is_input) {
    input_tensor_buffers_[buffer_index] = std::move(tensor_buffer);
    input_tensor_buffer_handles_[buffer_index] = buffer_handle;
    input_tensor_buffer_used_size_[buffer_index] = buffer_used_size;
  } else {
    output_tensor_buffers_[buffer_index] = std::move(tensor_buffer);
    output_tensor_buffer_handles_[buffer_index] = buffer_handle;
    output_tensor_buffer_used_size_[buffer_index] = buffer_used_size;
  }
  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Prepare(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::RegisterLiteRtTensorBuffers(
    TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  void* external_context;
  TfLiteOpaqueContextGetExternalContext(context, &external_context,
                                        kTfLiteLiteRtBufferContext);
  auto* buffer_context =
      reinterpret_cast<litert::internal::ExternalLiteRtBufferContext*>(
          external_context);

  size_t num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  for (size_t i = 0; i < num_node_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    auto tensor_buffer = buffer_context->GetTensorBuffer(tfl_opaque_tensor);
    if (tensor_buffer.HasValue()) {
      // TODO - b/379176766: If the provided TensorBuffer is not supported
      // types, we need to create a new one and convert the data from the
      // provided TensorBuffer.
      auto buffer_size = tensor_buffer->Size();
      if (!buffer_size) {
        LITERT_LOG(LITERT_ERROR, "%s", buffer_size.Error().Message().data());
        return kTfLiteError;
      }
      if (auto status = RegisterLiteRtTensorBuffer(std::move(*tensor_buffer),
                                                   *buffer_size, i,
                                                   /*is_input=*/true);
          status != kTfLiteOk) {
        return status;
      }
      input_tensor_buffers_require_cpu_sync_[i] = false;
    } else {
      LITERT_LOG(LITERT_INFO,
                 "Input#%d TensorBuffer is not registered. Create a new one",
                 i);
      if (auto status =
              CreateAndSetBuffer(tfl_opaque_tensor, i, /*is_input=*/true);
          status != kTfLiteOk) {
        return status;
      }
      input_tensor_buffers_require_cpu_sync_[i] = true;
    }
  }

  size_t num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  for (size_t i = 0; i < num_node_outputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    auto tensor_buffer = buffer_context->GetTensorBuffer(tfl_opaque_tensor);
    if (tensor_buffer.HasValue()) {
      // TODO - b/379176766: If the provided TensorBuffer is not supported
      // types, we need to create a new one and convert the data back to the
      // provided TensorBuffer.
      auto buffer_size = tensor_buffer->Size();
      if (!buffer_size) {
        LITERT_LOG(LITERT_ERROR, "%s", buffer_size.Error().Message().data());
        return kTfLiteError;
      }
      if (auto status = RegisterLiteRtTensorBuffer(std::move(*tensor_buffer),
                                                   *buffer_size, i,
                                                   /*is_input=*/false);
          status != kTfLiteOk) {
        return status;
      }
      output_tensor_buffers_require_cpu_sync_[i] = false;
    } else {
      LITERT_LOG(LITERT_INFO,
                 "Output#%d TensorBuffer is not registered. Create a new one",
                 i);
      if (auto status =
              CreateAndSetBuffer(tfl_opaque_tensor, i, /*is_input=*/false);
          status != kTfLiteOk) {
        return status;
      }
      output_tensor_buffers_require_cpu_sync_[i] = true;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                          TfLiteOpaqueNode* node) {
  if (auto status = RegisterLiteRtTensorBuffers(context, node);
      status != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to register tensor buffers: %d", status);
    return kTfLiteError;
  }

  size_t num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  if (num_node_inputs != input_tensor_buffers_.size()) {
    LITERT_LOG(LITERT_ERROR, "Invalid number of inputs");
    return kTfLiteError;
  }

  for (size_t i = 0; i < num_node_inputs; ++i) {
    if (!input_tensor_buffers_require_cpu_sync_[i]) {
      continue;
    }
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    void* tensor_data = TfLiteOpaqueTensorData(tfl_opaque_tensor);
    auto& tensor_buffer = input_tensor_buffers_[i];

    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    if (!lock_and_addr) {
      LITERT_LOG(LITERT_ERROR, "%s", lock_and_addr.Error().Message().data());
      return kTfLiteError;
    }

    size_t buffer_size = input_tensor_buffer_used_size_[i];
    std::memcpy(lock_and_addr->second, tensor_data, buffer_size);
  }

  if (auto status = LiteRtDispatchInvoke(invocation_context_);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke context: %d", status);
    return kTfLiteError;
  }

  size_t num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  if (num_node_outputs != output_tensor_buffers_.size()) {
    LITERT_LOG(LITERT_ERROR, "Invalid number of outputs");
    return kTfLiteError;
  }

  for (size_t i = 0; i < num_node_outputs; ++i) {
    if (!output_tensor_buffers_require_cpu_sync_[i]) {
      continue;
    }
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    void* tensor_data = TfLiteOpaqueTensorData(tfl_opaque_tensor);
    auto& tensor_buffer = output_tensor_buffers_[i];

    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    if (!lock_and_addr) {
      LITERT_LOG(LITERT_ERROR, "%s", lock_and_addr.Error().Message().data());
      return kTfLiteError;
    }

    size_t buffer_size = output_tensor_buffer_used_size_[i];
    std::memcpy(tensor_data, lock_and_addr->second, buffer_size);
  }

  return kTfLiteOk;
}

}  // namespace internal
}  // namespace litert
