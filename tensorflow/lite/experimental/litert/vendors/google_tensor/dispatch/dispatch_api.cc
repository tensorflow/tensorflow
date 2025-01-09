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

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/dispatch_api.h"

#include <cstdio>
#include <cstring>
#include <optional>
#include <set>
#include <string>

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif

#include "absl/strings/string_view.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch_api.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/southbound.h"

namespace {

constexpr const int VERSION_MAJOR = 0;
constexpr const int VERSION_MINOR = 1;
constexpr const int VERSION_PATCH = 0;

// We store THR names in a global set as a workaround to b/369144429.
std::set<std::string> ThrNames;

absl::string_view ThrNodeIdStr(LiteRtDispatchNodeId node_id) {
  auto str = "node_" + std::to_string(node_id);
  auto iter = ThrNames.find(str);
  if (iter == ThrNames.end()) {
    iter = ThrNames.insert(iter, str);
  }
  return *iter;
}

absl::string_view ThrEdgeIdStr(LiteRtDispatchEdgeId edge_id) {
  auto str = "edge_" + std::to_string(edge_id);
  auto iter = ThrNames.find(str);
  if (iter == ThrNames.end()) {
    iter = ThrNames.insert(iter, str);
  }
  return *iter;
}

litert::google_tensor::Southbound* TheSouthbound;
char BuildId[256];

}  // namespace

namespace litert {
namespace google_tensor {

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

  if (auto southbound =
          litert::google_tensor::Southbound::Create(shared_library_dir_opt);
      !southbound) {
    LITERT_LOG(LITERT_INFO, "Initialization failure: %s",
               southbound.Error().Message().data());
    return southbound.Error().Status();
  } else {
    TheSouthbound = southbound->release();
  }

  auto thr_initialize = TheSouthbound->api().thr_initialize;
  if (!thr_initialize) {
    LITERT_LOG(LITERT_INFO, "thr_initialize not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (auto status = thr_initialize(); status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_INFO, "thr_initialize failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  auto thr_get_vendor_api_version =
      TheSouthbound->api().thr_get_vendor_api_version;
  const char* sb_api_version =
      thr_get_vendor_api_version ? thr_get_vendor_api_version() : "N.A.";
  auto thr_get_vendor_id = TheSouthbound->api().thr_get_vendor_id;
  const char* sb_vendor_id = thr_get_vendor_id ? thr_get_vendor_id() : "N.A.";
  snprintf(
      BuildId, sizeof(BuildId),
      "GoogleTensor Dispatch API version %d.%d.%d, Darwinn API version %s, "
      "vendor id: %s",
      VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, sb_api_version,
      sb_vendor_id);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Google";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic |
                  kLiteRtDispatchCapabilitiesAsync |
                  kLiteRtDispatchCapabilitiesGraph;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  if (auto context = LiteRtDispatchDeviceContextT::Create(*TheSouthbound);
      context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().data());
    return context.Error().Status();
  }
  return kLiteRtStatusOk;
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
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status =
          LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get buffer type: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  if (tensor_buffer_type != kLiteRtTensorBufferTypeAhwb) {
    LITERT_LOG(LITERT_ERROR, "Unsupported buffer type: %d", tensor_buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get buffer size: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    if (status == kLiteRtStatusErrorNotFound) {
      tensor_buffer_offset = 0;
    } else {
      LITERT_LOG(LITERT_ERROR, "Failed to get buffer offset: %d", status);
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer type: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    LITERT_LOG(LITERT_ERROR, "Tensor strides are not supported");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  AHardwareBuffer* ahwb;
#if LITERT_HAS_AHWB_SUPPORT
  if (auto status = LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get AHWB: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }
#else
  LITERT_LOG(LITERT_ERROR, "AHardwareBuffer is not supported on this platform");
  return kLiteRtStatusErrorRuntimeFailure;
#endif

  ThrContext* thr_context = device_context->thr_context();
  ThrBufferHandle thr_buffer_handle;

  if (tensor_buffer_offset == 0) {
    auto thr_register_buffer = TheSouthbound->api().thr_register_buffer;
    if (!thr_register_buffer) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer not found");
      return kLiteRtStatusErrorRuntimeFailure;
    }

    if (auto status = thr_register_buffer(
            thr_context, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer failed: %d", status);
      return kLiteRtStatusErrorRuntimeFailure;
    }

  } else {
    auto thr_register_buffer_with_offset =
        TheSouthbound->api().thr_register_buffer_with_offset;
    if (!thr_register_buffer_with_offset) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer_with_offset not found");
      return kLiteRtStatusErrorRuntimeFailure;
    }

    if (auto status = thr_register_buffer_with_offset(
            thr_context, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_offset, tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer_with_offset failed: %d",
                 status);
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  *tensor_buffer_handle = thr_buffer_handle;
  return kLiteRtStatusOk;
}

LiteRtStatus UnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_unregister_buffer = TheSouthbound->api().thr_unregister_buffer;
  if (!thr_unregister_buffer) {
    LITERT_LOG(LITERT_ERROR, "thr_unregister_buffer not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_unregister_buffer(device_context->thr_context(),
                                          thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_register_buffer failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* exec_bytecode,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* invocation_context) {
  LiteRtDispatchGraph graph = nullptr;
  if (auto status = GraphCreate(device_context, &graph);
      status != kLiteRtStatusOk) {
    return status;
  }

  LiteRtDispatchNodeId node_id = 0;
  LiteRtDispatchNodeType node_type;
  switch (exec_type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      node_type = kLiteRtDispatchNodeTypeDsp;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      node_type = kLiteRtDispatchNodeTypeNpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected executable type: %d", exec_type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  if (auto status = AddNode(graph, node_id, node_type);
      status != kLiteRtStatusOk) {
    return status;
  }

  LiteRtDispatchExecutableHandle exec_handle;
  if (auto status = LoadExecutable(device_context, exec_type, exec_bytecode,
                                   exec_bytecode_size, &exec_handle);
      status != kLiteRtStatusOk) {
    return status;
  }

  if (auto status =
          AssignNodeFunction(graph, node_id, exec_handle, function_name);
      status != kLiteRtStatusOk) {
    return status;
  }

  LiteRtDispatchEdgeId next_edge_id = 0;

  for (auto input_index = 0; input_index < num_inputs; ++input_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = AddEdge(graph, edge_id); status != kLiteRtStatusOk) {
      return status;
    }
    if (auto status = ConnectGraphInput(graph, input_index, edge_id);
        status != kLiteRtStatusOk) {
      return status;
    }
    if (auto status = ConnectNodeInput(graph, node_id, input_index, edge_id);
        status != kLiteRtStatusOk) {
      return status;
    }
  }

  for (auto output_index = 0; output_index < num_outputs; ++output_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = AddEdge(graph, edge_id); status != kLiteRtStatusOk) {
      return status;
    }
    if (auto status = ConnectNodeOutput(graph, node_id, output_index, edge_id);
        status != kLiteRtStatusOk) {
      return status;
    }
    if (auto status = ConnectGraphOutput(graph, output_index, edge_id);
        status != kLiteRtStatusOk) {
      return status;
    }
  }

  if (auto status = InvocationContextCreateFromGraph(device_context, graph,
                                                     invocation_context);
      status != kLiteRtStatusOk) {
    return status;
  }

  (*invocation_context)->AttachExecutable(exec_handle);

  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_delete =
      TheSouthbound->api().thr_invocation_context_delete;
  if (!thr_invocation_context_delete) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_delete not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = invocation_context->graph()->thr_graph();
  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_delete(thr_graph, thr_icontext);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_delete failed: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  delete invocation_context;

  return kLiteRtStatusOk;
}

LiteRtStatus AttachBufferHelper(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_attach_buffer =
      TheSouthbound->api().thr_invocation_context_attach_buffer;
  if (!thr_invocation_context_attach_buffer) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_attach_buffer not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_attach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_attach_buffer failed: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status_or =
          invocation_context->graph()->InputEdge(graph_input_index);
      !status_or) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph input index: %d",
               graph_input_index);
    return kLiteRtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status_or;
    return AttachBufferHelper(invocation_context, edge_id,
                              tensor_buffer_handle);
  }
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->graph()->OutputEdge(graph_output_index);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph output index: %d",
               graph_output_index);
    return kLiteRtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status;
    return AttachBufferHelper(invocation_context, edge_id,
                              tensor_buffer_handle);
  }
}

LiteRtStatus DetachTensorBufferHelper(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_detach_buffer =
      TheSouthbound->api().thr_invocation_context_detach_buffer;
  if (!thr_invocation_context_detach_buffer) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_detach_buffer not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_detach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_detach_buffer failed: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status_or =
          invocation_context->graph()->InputEdge(graph_input_index);
      !status_or) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph input index: %d",
               graph_input_index);
    return kLiteRtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status_or;
    return DetachTensorBufferHelper(invocation_context, edge_id,
                                    tensor_buffer_handle);
  }
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->graph()->OutputEdge(graph_output_index);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph output index: %d",
               graph_output_index);
    return kLiteRtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status;
    return DetachTensorBufferHelper(invocation_context, edge_id,
                                    tensor_buffer_handle);
  }
}

LiteRtStatus PrepareForInvoke(
    LiteRtDispatchInvocationContext invocation_context,
    bool create_output_sync_fence) {
  auto thr_invocation_context_prepare_for_invoke =
      TheSouthbound->api().thr_invocation_context_prepare_for_invoke;
  if (!thr_invocation_context_prepare_for_invoke) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_prepare_for_invoke not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_prepare_for_invoke(
          thr_icontext, create_output_sync_fence);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_prepare_for_invoke failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus InvokeOnce(LiteRtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_invoke_once =
      TheSouthbound->api().thr_invocation_context_invoke_once;
  if (!thr_invocation_context_invoke_once) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_invoke_once not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_invoke_once(thr_icontext);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_invoke_once failed: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus Wait(LiteRtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_wait =
      TheSouthbound->api().thr_invocation_context_wait;
  if (!thr_invocation_context_wait) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_wait not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_wait(thr_icontext);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_wait failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = PrepareForInvoke(invocation_context,
                                     /*create_output_sync_fence=*/false);
      status != kLiteRtStatusOk) {
    return status;
  }

  if (auto status = InvokeOnce(invocation_context); status != kLiteRtStatusOk) {
    return status;
  }
  return Wait(invocation_context);
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus AttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  auto status_or = invocation_context->graph()->InputEdge(graph_input_index);
  if (!status_or) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph input index: %d",
               graph_input_index);
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto edge_id = *status_or;

  auto thr_invocation_context_attach_input_buffer_sync_fence =
      TheSouthbound->api()
          .thr_invocation_context_attach_input_buffer_sync_fence;
  if (!thr_invocation_context_attach_input_buffer_sync_fence) {
    LITERT_LOG(
        LITERT_ERROR,
        "thr_invocation_context_attach_input_buffer_sync_fence not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  int input_fence_fd;
  if (auto status = LiteRtGetEventSyncFenceFd(input_event, &input_fence_fd);
      status != kLiteRtStatusOk) {
    return status;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_invocation_context_attach_input_buffer_sync_fence(
          thr_icontext, thr_edge_id.data(), input_fence_fd);
      status != kThrStatusSuccess) {
    LITERT_LOG(
        LITERT_ERROR,
        "thr_invocation_context_attach_input_buffer_sync_fence failed: %d",
        status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus GetOutputEvent(LiteRtDispatchInvocationContext invocation_context,
                            int graph_output_index, LiteRtEvent* output_event) {
  auto status_or = invocation_context->graph()->OutputEdge(graph_output_index);
  if (!status_or) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph output index: %d",
               graph_output_index);
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto edge_id = *status_or;

  auto thr_invocation_context_get_output_buffer_sync_fence =
      TheSouthbound->api().thr_invocation_context_get_output_buffer_sync_fence;
  if (!thr_invocation_context_get_output_buffer_sync_fence) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_get_output_buffer_sync_fence not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  int output_fence_fd;
  if (auto status = thr_invocation_context_get_output_buffer_sync_fence(
          thr_icontext, thr_edge_id.data(), &output_fence_fd);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_get_output_buffer_sync_fence failed: %d",
               status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  if (auto status = LiteRtCreateEventFromSyncFenceFd(
          output_fence_fd, /*owns_fd=*/false, output_event);
      status != kLiteRtStatusOk) {
    return status;
  }

  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus InvokeAsync(LiteRtDispatchInvocationContext invocation_context,
                         int num_output_events, LiteRtEvent* output_events) {
  if (num_output_events != invocation_context->graph()->NumOutputs()) {
    LITERT_LOG(LITERT_ERROR, "Unexpected number of output events: %d",
               num_output_events);
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (auto status = PrepareForInvoke(invocation_context,
                                     /*create_output_sync_fence=*/true);
      status != kLiteRtStatusOk) {
    return status;
  }

  if (auto status = InvokeOnce(invocation_context); status != kLiteRtStatusOk) {
    return status;
  }

  for (auto graph_output_index = 0; graph_output_index < num_output_events;
       ++graph_output_index) {
    if (auto status = GetOutputEvent(invocation_context, graph_output_index,
                                     &output_events[graph_output_index]);
        status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to get event for output %d: %d",
                 graph_output_index, status);
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus GraphCreate(LiteRtDispatchDeviceContext device_context,
                         LiteRtDispatchGraph* graph) {
  auto thr_graph_create = TheSouthbound->api().thr_graph_create;
  if (!thr_graph_create) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_create not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = thr_graph_create(device_context->thr_context());
  if (!thr_graph) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_create failed");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *graph = new LiteRtDispatchGraphT(thr_graph, device_context);
  return kLiteRtStatusOk;
}

LiteRtStatus GraphDestroy(LiteRtDispatchGraph graph) {
  auto thr_graph_delete = TheSouthbound->api().thr_graph_delete;
  if (!thr_graph_delete) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_delete not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  graph->device_context()->remove_graph(graph->thr_graph());

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thr_graph_delete(thr_graph); status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_destroy failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  delete graph;
  return kLiteRtStatusOk;
}

LiteRtStatus AddNode(LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
                     LiteRtDispatchNodeType node_type) {
  auto thr_graph_add_sq_node = TheSouthbound->api().thr_graph_add_sq_node;
  if (!thr_graph_add_sq_node) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_sq_node not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrNodeType thr_node_type;
  switch (node_type) {
    case kLiteRtDispatchNodeTypeDsp:
      thr_node_type = kThrNodeTypeDsp;
      break;
    case kLiteRtDispatchNodeTypeNpu:
      thr_node_type = kThrNodeTypeNpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected node type: %d", node_type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  if (auto status =
          thr_graph_add_sq_node(thr_graph, thr_node_id.data(), thr_node_type);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_sq_node failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AddEdge(LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_add_edge = TheSouthbound->api().thr_graph_add_edge;
  if (!thr_graph_add_edge) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_edge not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrEdgeType thr_edge_type = kThrEdgeNoType;
  if (auto status =
          thr_graph_add_edge(thr_graph, thr_edge_id.data(), thr_edge_type);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_edge failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ConnectNodeInput(LiteRtDispatchGraph graph,
                              LiteRtDispatchNodeId node_id, int input_index,
                              LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_input =
      TheSouthbound->api().thr_graph_connect_node_input;
  if (!thr_graph_connect_node_input) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_connect_node_input not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  int next_input_index = graph->NextNodeInputIndex(node_id);
  if (input_index != next_input_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected input index %d, expected %d",
               input_index, next_input_index);
    return kLiteRtStatusErrorInvalidArgument;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_input(thr_graph, thr_node_id.data(),
                                                 thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_input_edge failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  graph->AddInputEdge(input_index, edge_id);
  return kLiteRtStatusOk;
}

LiteRtStatus ConnectNodeOutput(LiteRtDispatchGraph graph,
                               LiteRtDispatchNodeId node_id, int output_index,
                               LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_output =
      TheSouthbound->api().thr_graph_connect_node_output;
  if (!thr_graph_connect_node_output) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_connect_node_output not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  int next_output_index = graph->NextNodeOutputIndex(node_id);
  if (output_index != next_output_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected output index %d, expected %d",
               output_index, next_output_index);
    return kLiteRtStatusErrorInvalidArgument;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_output(thr_graph, thr_node_id.data(),
                                                  thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_output_edge failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  graph->AddOutputEdge(output_index, edge_id);
  return kLiteRtStatusOk;
}

LiteRtStatus ConnectGraphInput(LiteRtDispatchGraph graph, int input_index,
                               LiteRtDispatchEdgeId edge_id) {
  int next_input_index = graph->NextGraphInputIndex();
  if (input_index != next_input_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected input index %d, expected %d",
               input_index, next_input_index);
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto thr_graph_set_input_edge = TheSouthbound->api().thr_graph_set_input_edge;
  if (!thr_graph_set_input_edge) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_input_edge not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_input_edge(thr_graph, thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_input_edge failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ConnectGraphOutput(LiteRtDispatchGraph graph, int output_index,
                                LiteRtDispatchEdgeId edge_id) {
  int next_output_index = graph->NextGraphOutputIndex();
  if (output_index != next_output_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected output index %d, expected %d",
               output_index, next_output_index);
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto thr_graph_set_output_edge =
      TheSouthbound->api().thr_graph_set_output_edge;
  if (!thr_graph_set_output_edge) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_output_edge not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_output_edge(thr_graph, thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_output_edge failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LoadExecutable(LiteRtDispatchDeviceContext device_context,
                            LiteRtDispatchExecutableType type,
                            const void* bytecode, size_t bytecode_size,
                            LiteRtDispatchExecutableHandle* exec_handle) {
  auto thr_load_sq_container = TheSouthbound->api().thr_load_sq_container;
  if (!thr_load_sq_container) {
    LITERT_LOG(LITERT_ERROR, "thr_load_sq_container not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrSqContainerType thr_type;
  switch (type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      thr_type = kThrSqContainerTypeFunctionLibrary;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      thr_type = kThrSqContainerTypeMlModel;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected executable type: %d", type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  ThrContext* thr_context = device_context->thr_context();
  ThrSqContainerHandle sq_handle;
  if (auto status = thr_load_sq_container(thr_context, thr_type, bytecode,
                                          bytecode_size, &sq_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_load_sq_container failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *exec_handle = sq_handle;
  return kLiteRtStatusOk;
}

LiteRtStatus UnloadExecutable(LiteRtDispatchDeviceContext device_context,
                              LiteRtDispatchExecutableHandle exec_handle) {
  auto thr_unload_sq_container = TheSouthbound->api().thr_unload_sq_container;
  if (!thr_unload_sq_container) {
    LITERT_LOG(LITERT_ERROR, "thr_unload_sq_container not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrContext* thr_context = device_context->thr_context();
  ThrSqContainerHandle sq_handle = exec_handle;
  if (auto status = thr_unload_sq_container(thr_context, sq_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_unload_sq_container failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AssignNodeFunction(LiteRtDispatchGraph graph,
                                LiteRtDispatchNodeId node_id,
                                LiteRtDispatchExecutableHandle exec_handle,
                                const char* function_name) {
  auto thr_graph_assign_sq = TheSouthbound->api().thr_graph_assign_sq;
  if (!thr_graph_assign_sq) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_assign_sq not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrSqContainerHandle sq_handle = exec_handle;
  // An empty function name represent no function name being provided and
  // therefore we must pass a nullptr to the call below, otherwise the SB API
  // will expect a model with a signature. See b/378913220.
  const char* function_name_ptr =
      absl::string_view(function_name).empty() ? nullptr : function_name;
  if (auto status = thr_graph_assign_sq(thr_graph, thr_node_id.data(),
                                        sq_handle, function_name_ptr);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_assign_sq failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AnnotateGraph(LiteRtDispatchGraph graph, const char* key,
                           const char* value) {
  auto thr_graph_annotate_graph = TheSouthbound->api().thr_graph_annotate_graph;
  if (!thr_graph_annotate_graph) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_graph not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thr_graph_annotate_graph(thr_graph, key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_graph failed");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AnnotateNode(LiteRtDispatchGraph graph,
                          LiteRtDispatchNodeId node_id, const char* key,
                          const char* value) {
  auto thr_graph_annotate_node = TheSouthbound->api().thr_graph_annotate_node;
  if (!thr_graph_annotate_node) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_node not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  if (auto status =
          thr_graph_annotate_node(thr_graph, thr_node_id.data(), key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_node failed");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AnnotateEdge(LiteRtDispatchGraph graph,
                          LiteRtDispatchEdgeId edge_id, const char* key,
                          const char* value) {
  auto thr_graph_annotate_edge = TheSouthbound->api().thr_graph_annotate_edge;
  if (!thr_graph_annotate_edge) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_edge not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status =
          thr_graph_annotate_edge(thr_graph, thr_edge_id.data(), key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_edge failed");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  auto thr_invocation_context_get =
      TheSouthbound->api().thr_invocation_context_get;
  if (!thr_invocation_context_get) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_get not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_icontext =
      thr_invocation_context_get(thr_graph, device_context->thr_context());
  if (!thr_icontext) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_get failed");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  device_context->add_graph(thr_graph);
  *invocation_context =
      new LiteRtDispatchInvocationContextT(thr_icontext, device_context, graph);

  return kLiteRtStatusOk;
}

}  // namespace google_tensor
}  // namespace litert

// /////////////////////////////////////////////////////////////////////////////

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::google_tensor::Initialize,
    .get_vendor_id = litert::google_tensor::GetVendorId,
    .get_build_id = litert::google_tensor::GetBuildId,
    .get_capabilities = litert::google_tensor::GetCapabilities,
    .device_context_create = litert::google_tensor::DeviceContextCreate,
    .device_context_destroy = litert::google_tensor::DeviceContextDestroy,
    .get_input_requirements = litert::google_tensor::GetInputRequirements,
    .get_output_requirements = litert::google_tensor::GetOutputRequirements,
    .register_tensor_buffer = litert::google_tensor::RegisterTensorBuffer,
    .unregister_tensor_buffer = litert::google_tensor::UnregisterTensorBuffer,
    .invocation_context_create = litert::google_tensor::InvocationContextCreate,
    .invocation_context_destroy =
        litert::google_tensor::InvocationContextDestroy,
    .attach_input = litert::google_tensor::AttachInput,
    .attach_output = litert::google_tensor::AttachOutput,
    .detach_input = litert::google_tensor::DetachInput,
    .detach_output = litert::google_tensor::DetachOutput,
    .invoke = litert::google_tensor::Invoke,
};

LiteRtDispatchAsyncInterface TheAsyncInterface = {
    .attach_input_event = litert::google_tensor::AttachInputEvent,
    .invoke_async = litert::google_tensor::InvokeAsync,
};

LiteRtDispatchGraphInterface TheGraphInterface = {
    .graph_create = litert::google_tensor::GraphCreate,
    .graph_destroy = litert::google_tensor::GraphDestroy,
    .add_node = litert::google_tensor::AddNode,
    .add_edge = litert::google_tensor::AddEdge,
    .connect_node_input = litert::google_tensor::ConnectNodeInput,
    .connect_node_output = litert::google_tensor::ConnectNodeOutput,
    .connect_graph_input = litert::google_tensor::ConnectGraphInput,
    .connect_graph_output = litert::google_tensor::ConnectGraphOutput,
    .load_executable = litert::google_tensor::LoadExecutable,
    .unload_executable = litert::google_tensor::UnloadExecutable,
    .assign_node_function = litert::google_tensor::AssignNodeFunction,
    .annotate_graph = litert::google_tensor::AnnotateGraph,
    .annotate_node = litert::google_tensor::AnnotateNode,
    .annotate_edge = litert::google_tensor::AnnotateEdge,
    .invocation_context_create_from_graph =
        litert::google_tensor::InvocationContextCreateFromGraph,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = VERSION_MAJOR,
                .minor = VERSION_MINOR,
                .patch = VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = &TheAsyncInterface,
    .graph_interface = &TheGraphInterface,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
