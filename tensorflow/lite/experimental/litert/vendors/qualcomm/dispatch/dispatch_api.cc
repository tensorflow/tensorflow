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

#include <memory>
#include <optional>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch_api.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace {

using ::litert::qnn::QnnManager;

static constexpr const int VERSION_MAJOR = 0;
static constexpr const int VERSION_MINOR = 1;
static constexpr const int VERSION_PATCH = 0;

static std::unique_ptr<QnnManager> TheQnnManager;

QnnManager& Qnn() { return *TheQnnManager; }

char BuildId[256];

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

const char* GetSharedLibraryDir(const LiteRtDispatchOption* options,
                                int num_options) {
  for (auto i = 0; i < num_options; ++i) {
    auto& option = options[i];
    if (!strcmp(option.name, kDispatchOptionSharedLibraryDir)) {
      return option.value.str_value;
    }
  }
  return nullptr;
}

LiteRtStatus Initialize(const LiteRtDispatchOption* options, int num_options) {
  auto* shared_library_dir = GetSharedLibraryDir(options, num_options);
  std::optional<std::string> shared_library_dir_opt =
      shared_library_dir ? std::make_optional(std::string(shared_library_dir))
                         : std::nullopt;

  auto configs = QnnManager::DefaultBackendConfigs();
  if (auto qnn_manager = QnnManager::Create(configs, shared_library_dir_opt);
      !qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().data());
    return qnn_manager.Error().Status();
  } else {
    std::swap(TheQnnManager, *qnn_manager);
  }

  Qnn_ApiVersion_t qnn_api_version;
  if (auto status = Qnn().Api()->backendGetApiVersion(&qnn_api_version);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN API version: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  const char* build_id;
  if (auto status = Qnn().Api()->backendGetBuildId(&build_id);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN build ID: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  snprintf(BuildId, sizeof(BuildId),
           "Qualcomm Dispatch API version %d.%d.%d, QNN API version %d.%d.%d, "
           "build id: %s",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
           qnn_api_version.coreApiVersion.major,
           qnn_api_version.coreApiVersion.minor,
           qnn_api_version.coreApiVersion.patch, build_id);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Qualcomm";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  if (auto context = LiteRtDispatchDeviceContextT::Create(Qnn()); context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().data());
    return context.Error().Status();
  }
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
               requirements.Error().Message().data());
    return requirements.Error().Status();
  }
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
               requirements.Error().Message().data());
    return requirements.Error().Status();
  }
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (auto status = device_context->RegisterTensorBuffer(buffer); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to register buffer: %s",
               status.Error().Message().data());
    return status.Error().Status();
  } else {
    *tensor_buffer_handle = *status;
    return kLiteRtStatusOk;
  }
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  if (auto status = device_context->UnregisterTensorBuffer(handle); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister buffer: %s",
               status.Error().Message().data());
    return status.Error().Status();
  } else {
    return kLiteRtStatusOk;
  }
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      Qnn(), *device_context, exec_bytecode_ptr, exec_bytecode_size,
      function_name);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create context from context binary: %s",
               context.Error().Message().data());
    return context.Error().Status();
  }
  *invocation_context = context->release();
  device_context->SetInvocationContext(*invocation_context);
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input buffer: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach output buffer: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do here.
  return kLiteRtStatusOk;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do here.
  return kLiteRtStatusOk;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Execute(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to execute invocation context: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////

LiteRtDispatchInterface TheInterface = {
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/GetInputRequirements,
    /*.get_output_requirements=*/GetOutputRequirements,
    /*.register_tensor_buffer=*/RegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/UnregisterTensorBuffer,
    /*.invocation_context_create=*/InvocationContextCreate,
    /*.invocation_context_destroy=*/InvocationContextDestroy,
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
};

LiteRtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/VERSION_MAJOR,
                  /*.minor=*/VERSION_MINOR,
                  /*.patch=*/VERSION_PATCH},
    /*.interface=*/&TheInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
