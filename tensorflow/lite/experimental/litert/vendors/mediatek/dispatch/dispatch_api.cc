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

#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch_api.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_invocation_context.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

namespace {

litert::mediatek::NeuronAdapter* TheNeuronAdapter;
char BuildId[256];

}  // namespace

namespace litert {
namespace mediatek {

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

LiteRtStatus LiteRtInitialize(const LiteRtDispatchOption* options,
                              int num_options) {
  auto* shared_library_dir = GetSharedLibraryDir(options, num_options);
  std::optional<std::string> shared_library_dir_opt =
      shared_library_dir ? std::make_optional(std::string(shared_library_dir))
                         : std::nullopt;

  if (auto neuron_adapter =
          litert::mediatek::NeuronAdapter::Create(shared_library_dir_opt);
      neuron_adapter) {
    TheNeuronAdapter = neuron_adapter->release();
  } else {
    LITERT_LOG(LITERT_INFO, "Initialization failure: %s",
               neuron_adapter.Error().Message().data());
    return neuron_adapter.Error().Status();
  }

  auto get_version = TheNeuronAdapter->api().get_version;
  if (!get_version) {
    LITERT_LOG(LITERT_ERROR, "get_version not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  NeuronRuntimeVersion version;
  if (get_version(&version) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to get version");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LITERT_LOG(LITERT_INFO, "Neuron SDK version: %d.%d.%d", version.major,
             version.minor, version.patch);

  snprintf(BuildId, sizeof(BuildId),
           "MediaTek Dispatch API version %d.%d.%d, NeuronAdaptor API version "
           "%d.%d.%d",
           LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
           LITERT_API_VERSION_PATCH, version.major, version.minor,
           version.patch);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetVendorId(const char** vendor_id) {
  *vendor_id = "MediaTek";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  if (auto context = LiteRtDispatchDeviceContextT::Create(*TheNeuronAdapter);
      context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().data());
    return context.Error().Status();
  }
}

LiteRtStatus LiteRtDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetInputRequirements(
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

LiteRtStatus LiteRtGetOutputRequirements(
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

LiteRtStatus LiteRtRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  litert::TensorBuffer tensor_buffer_(tensor_buffer, /*owned=*/false);
  if (auto result = device_context->RegisterTensorBuffer(tensor_buffer_);
      result) {
    *tensor_buffer_handle = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to register tensor buffer: %s",
               result.Error().Message().data());
    return result.Error().Status();
  }
}

LiteRtStatus LiteRtUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status =
          device_context->UnregisterTensorBuffer(tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister tensor buffer: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      *TheNeuronAdapter, device_context, exec_type, exec_bytecode_ptr,
      exec_bytecode_size, function_name, num_inputs, num_outputs);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create context from context binary: %s",
               context.Error().Message().data());
    return context.Error().Status();
  }
  *invocation_context = context->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach output: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->DetachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to detach input: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->DetachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to detach output: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtInvoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Invoke(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke context: %s",
               status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // namespace mediatek
}  // namespace litert

// /////////////////////////////////////////////////////////////////////////////

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::mediatek::LiteRtInitialize,
    .get_vendor_id = litert::mediatek::LiteRtGetVendorId,
    .get_build_id = litert::mediatek::LiteRtGetBuildId,
    .get_capabilities = litert::mediatek::LiteRtGetCapabilities,
    .device_context_create = litert::mediatek::LiteRtDeviceContextCreate,
    .device_context_destroy = litert::mediatek::LiteRtDeviceContextDestroy,
    .get_input_requirements = litert::mediatek::LiteRtGetInputRequirements,
    .get_output_requirements = litert::mediatek::LiteRtGetOutputRequirements,
    .register_tensor_buffer = litert::mediatek::LiteRtRegisterTensorBuffer,
    .unregister_tensor_buffer = litert::mediatek::LiteRtUnregisterTensorBuffer,
    .invocation_context_create =
        litert::mediatek::LiteRtInvocationContextCreate,
    .invocation_context_destroy =
        litert::mediatek::LiteRtInvocationContextDestroy,
    .attach_input = litert::mediatek::LiteRtAttachInput,
    .attach_output = litert::mediatek::LiteRtAttachOutput,
    .detach_input = litert::mediatek::LiteRtDetachInput,
    .detach_output = litert::mediatek::LiteRtDetachOutput,
    .invoke = litert::mediatek::LiteRtInvoke,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = nullptr,
    .graph_interface = nullptr,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
