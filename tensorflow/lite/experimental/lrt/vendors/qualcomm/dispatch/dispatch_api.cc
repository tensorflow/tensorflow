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

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch/lrt_dispatch_device_context.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch/lrt_dispatch_invocation_context.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

namespace {

using ::lrt::qnn::QnnManager;
using ::lrt::qnn::SetupAll;

static constexpr const int VERSION_MAJOR = 0;
static constexpr const int VERSION_MINOR = 1;
static constexpr const int VERSION_PATCH = 0;

QnnManager& Qnn() {
  static QnnManager qnn_manager;
  return qnn_manager;
}

char BuildId[256];

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus Initialize() {
  LRT_RETURN_STATUS_IF_NOT_OK(SetupAll(/*soc_model=*/std::nullopt, Qnn(),
                                       /*load_system=*/true,
                                       /*load_context=*/false));

  Qnn_ApiVersion_t qnn_api_version;
  if (auto status = Qnn().Api()->backendGetApiVersion(&qnn_api_version);
      status != QNN_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to get QNN API version: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  const char* build_id;
  if (auto status = Qnn().Api()->backendGetBuildId(&build_id);
      status != QNN_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to get QNN build ID: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  snprintf(BuildId, sizeof(BuildId),
           "Qualcomm Dispatch API version %d.%d.%d, QNN API version %d.%d.%d, "
           "build id: %s",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
           qnn_api_version.coreApiVersion.major,
           qnn_api_version.coreApiVersion.minor,
           qnn_api_version.coreApiVersion.patch, build_id);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLrtStatusOk;
}

LrtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Qualcomm";
  return kLrtStatusOk;
}

LrtStatus GetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLrtStatusOk;
}

LrtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLrtDispatchCapabilitiesBasic;
  return kLrtStatusOk;
}

LrtStatus DeviceContextCreate(LrtDispatchDeviceContext* device_context) {
  if (auto status_or = LrtDispatchDeviceContextT::Create(Qnn());
      status_or.ok()) {
    *device_context = status_or->release();
    return kLrtStatusOk;
  } else {
    ABSL_LOG(ERROR) << "Failed to create device context: "
                    << status_or.status();
    return kLrtStatusErrorRuntimeFailure;
  }
}

LrtStatus DeviceContextDestroy(LrtDispatchDeviceContext device_context) {
  delete device_context;
  return kLrtStatusOk;
}

LrtStatus GetInputRequirements(
    LrtDispatchInvocationContext invocation_context, int input_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      requirements.ok()) {
    *tensor_buffer_requirements = *requirements;
    return kLrtStatusOk;
  } else {
    ABSL_LOG(ERROR) << "Failed to get tensor buffer requirements: "
                    << requirements.status();
    return kLrtStatusErrorRuntimeFailure;
  }
}

LrtStatus GetOutputRequirements(
    LrtDispatchInvocationContext invocation_context, int output_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      requirements.ok()) {
    *tensor_buffer_requirements = *requirements;
    return kLrtStatusOk;
  } else {
    ABSL_LOG(ERROR) << "Failed to get tensor buffer requirements: "
                    << requirements.status();
    return kLrtStatusErrorRuntimeFailure;
  }
}

LrtStatus RegisterTensorBuffer(LrtDispatchDeviceContext device_context,
                               LrtTensorBuffer buffer,
                               LrtTensorBufferHandle* tensor_buffer_handle) {
  if (auto status = device_context->RegisterTensorBuffer(buffer);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to register buffer: " << status;
    return kLrtStatusErrorRuntimeFailure;
  } else {
    *tensor_buffer_handle = *status;
    return kLrtStatusOk;
  }
}

LrtStatus UnregisterTensorBuffer(LrtDispatchDeviceContext device_context,
                                 LrtTensorBufferHandle handle) {
  if (auto status = device_context->UnregisterTensorBuffer(handle);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to unregister buffer: " << status;
    return kLrtStatusErrorRuntimeFailure;
  } else {
    return kLrtStatusOk;
  }
}

LrtStatus InvocationContextCreate(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LrtDispatchInvocationContext* invocation_context) {
  auto context = LrtDispatchInvocationContextT::Create(
      Qnn(), *device_context, exec_bytecode_ptr, exec_bytecode_size,
      function_name);
  if (!context.ok()) {
    ABSL_LOG(ERROR) << "Failed to create context from context binary: "
                    << context.status();
    return kLrtStatusErrorRuntimeFailure;
  }
  *invocation_context = context->release();
  return kLrtStatusOk;
}

LrtStatus InvocationContextDestroy(
    LrtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLrtStatusOk;
}

LrtStatus AttachInput(LrtDispatchInvocationContext invocation_context,
                      int graph_input_index,
                      LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to attach input buffer: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }
  return kLrtStatusOk;
}

LrtStatus AttachOutput(LrtDispatchInvocationContext invocation_context,
                       int graph_output_index,
                       LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to attach output buffer: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }
  return kLrtStatusOk;
}

LrtStatus DetachInput(LrtDispatchInvocationContext invocation_context,
                      int graph_input_index,
                      LrtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do here.
  return kLrtStatusOk;
}

LrtStatus DetachOutput(LrtDispatchInvocationContext invocation_context,
                       int graph_output_index,
                       LrtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do here.
  return kLrtStatusOk;
}

LrtStatus Invoke(LrtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Execute(); !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to execute invocation context: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }
  return kLrtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////

LrtDispatchInterface TheInterface = {
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

LrtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/VERSION_MAJOR,
                  /*.minor=*/VERSION_MINOR,
                  /*.patch=*/VERSION_PATCH},
    /*.interface=*/&TheInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace

LrtStatus LrtDispatchGetApi(LrtDispatchApi* api) {
  *api = TheApi;
  return kLrtStatusOk;
}
