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

#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/dispatch_api.h"

#include <dlfcn.h>
#include <poll.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/lrt_dispatch_device_context.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/lrt_dispatch_graph.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/lrt_dispatch_invocation_context.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/southbound.h"

namespace {

constexpr const int VERSION_MAJOR = 0;
constexpr const int VERSION_MINOR = 1;
constexpr const int VERSION_PATCH = 0;

constexpr char kDynamicInteropKey[] = "dynamic_interop_mode";
constexpr char kDisableEarlyWakeup[] = "1";
constexpr char kEnableEarlyWakeup[] = "2";

std::set<std::string> ThrNames;

absl::string_view ThrNodeIdStr(LrtDispatchNodeId node_id) {
  auto str = "node_" + std::to_string(node_id);
  auto iter = ThrNames.find(str);
  if (iter == ThrNames.end()) {
    iter = ThrNames.insert(iter, str);
  }
  return *iter;
}

absl::string_view ThrEdgeIdStr(LrtDispatchEdgeId edge_id) {
  auto str = "edge_" + std::to_string(edge_id);
  auto iter = ThrNames.find(str);
  if (iter == ThrNames.end()) {
    iter = ThrNames.insert(iter, str);
  }
  return *iter;
}

lrt::pixel::Southbound* TheSouthbound;
char BuildId[256];

}  // namespace

namespace lrt {
namespace pixel {

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus Initialize() {
  if (auto status = lrt::pixel::Southbound::Create(); !status.ok()) {
    ABSL_LOG(ERROR) << "Initialization failure: " << status;
    return kLrtStatusErrorRuntimeFailure;
  } else {
    TheSouthbound = status->release();
  }

  auto thr_initialize = TheSouthbound->thr_functions().thr_initialize;
  if (!thr_initialize) {
    ABSL_LOG(ERROR) << "thr_initialize not found";
    return kLrtStatusErrorRuntimeFailure;
  }
  if (auto status = thr_initialize(); status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_initialize failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  auto thr_get_vendor_api_version =
      TheSouthbound->thr_functions().thr_get_vendor_api_version;
  const char* sb_api_version =
      thr_get_vendor_api_version ? thr_get_vendor_api_version() : "N.A.";
  auto thr_get_vendor_id = TheSouthbound->thr_functions().thr_get_vendor_id;
  const char* sb_vendor_id = thr_get_vendor_id ? thr_get_vendor_id() : "N.A.";
  snprintf(BuildId, sizeof(BuildId),
           "Pixel Dispatch API version %d.%d.%d, Darwinn API version %s, "
           "vendor id: %s",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, sb_api_version,
           sb_vendor_id);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLrtStatusOk;
}

LrtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Google";
  return kLrtStatusOk;
}

LrtStatus GetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLrtStatusOk;
}

LrtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLrtDispatchCapabilitiesBasic |
                  kLrtDispatchCapabilitiesAsync | kLrtDispatchCapabilitiesGraph;
  return kLrtStatusOk;
}

LrtStatus DeviceContextCreate(LrtDispatchDeviceContext* device_context) {
  if (auto status_or = LrtDispatchDeviceContextT::Create(*TheSouthbound);
      status_or.ok()) {
    *device_context = status_or->release();
    return kLrtStatusOk;
  } else {
    ABSL_LOG(ERROR) << "Failed to create device context: "
                    << status_or.status();
    return kLrtStatusErrorRuntimeFailure;
  }
  return kLrtStatusOk;
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
                               LrtTensorBuffer tensor_buffer,
                               LrtTensorBufferHandle* tensor_buffer_handle) {
  LrtTensorBufferType tensor_buffer_type;
  if (auto status = LrtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLrtStatusOk) {
    ABSL_LOG(ERROR) << "Failed to get buffer type: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  if (tensor_buffer_type != kLrtTensorBufferTypeAhwb) {
    ABSL_LOG(ERROR) << "Unsupported buffer type: " << tensor_buffer_type;
    return kLrtStatusErrorUnsupported;
  }

  size_t tensor_buffer_size;
  if (auto status = LrtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLrtStatusOk) {
    ABSL_LOG(ERROR) << "Failed to get buffer size: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  size_t tensor_buffer_offset;
  if (auto status =
          LrtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLrtStatusOk) {
    if (status == kLrtStatusErrorNotFound) {
      tensor_buffer_offset = 0;
    } else {
      ABSL_LOG(ERROR) << "Failed to get buffer offset: " << status;
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  LrtRankedTensorType tensor_type;
  if (auto status = LrtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLrtStatusOk) {
    ABSL_LOG(ERROR) << "Failed to get tensor buffer's type: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    ABSL_LOG(ERROR) << "Tensor strides are not supported by Pixel";
    return kLrtStatusErrorRuntimeFailure;
  }

  AHardwareBuffer* ahwb;
#if LRT_HAS_AHWB_SUPPORT
  if (auto status = LrtGetTensorBufferAhwb(tensor_buffer, &ahwb);
      status != kLrtStatusOk) {
    ABSL_LOG(ERROR) << "Failed to get AHWB: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }
#else
  ABSL_LOG(ERROR) << "AHardwareBuffer is not supported on this platform";
  return kLrtStatusErrorRuntimeFailure;
#endif

  ThrContext* thr_context = device_context->thr_context();
  ThrBufferHandle thr_buffer_handle;

  if (tensor_buffer_offset == 0) {
    auto thr_register_buffer =
        TheSouthbound->thr_functions().thr_register_buffer;
    if (!thr_register_buffer) {
      ABSL_LOG(ERROR) << "thr_register_buffer not found";
      return kLrtStatusErrorRuntimeFailure;
    }

    if (auto status = thr_register_buffer(
            thr_context, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      ABSL_LOG(ERROR) << "thr_register_buffer failed: " << status;
      return kLrtStatusErrorRuntimeFailure;
    }

  } else {
    auto thr_register_buffer_with_offset =
        TheSouthbound->thr_functions().thr_register_buffer_with_offset;
    if (!thr_register_buffer_with_offset) {
      ABSL_LOG(ERROR) << "thr_register_buffer_with_offset not found";
      return kLrtStatusErrorRuntimeFailure;
    }

    if (auto status = thr_register_buffer_with_offset(
            thr_context, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_offset, tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      ABSL_LOG(ERROR) << "thr_register_buffer_with_offset failed: " << status;
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  *tensor_buffer_handle = thr_buffer_handle;
  return kLrtStatusOk;
}

LrtStatus UnregisterTensorBuffer(LrtDispatchDeviceContext device_context,
                                 LrtTensorBufferHandle tensor_buffer_handle) {
  auto thr_unregister_buffer =
      TheSouthbound->thr_functions().thr_unregister_buffer;
  if (!thr_unregister_buffer) {
    ABSL_LOG(ERROR) << "thr_unregister_buffer not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_unregister_buffer(device_context->thr_context(),
                                          thr_buffer_handle);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_unregister_buffer failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus InvocationContextCreate(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableType exec_type, const void* exec_bytecode,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LrtDispatchInvocationContext* invocation_context) {
  LrtDispatchGraph graph = nullptr;
  if (auto status = GraphCreate(device_context, &graph);
      status != kLrtStatusOk) {
    return status;
  }

  if (auto status =
          AnnotateGraph(graph, kDynamicInteropKey, kEnableEarlyWakeup);
      status != kLrtStatusOk) {
    return status;
  }

  LrtDispatchNodeId node_id = 0;
  LrtDispatchNodeType node_type;
  switch (exec_type) {
    case kLrtDispatchExecutableTypeDspLibrary:
      node_type = kLrtDispatchNodeTypeDsp;
      break;
    case kLrtDispatchExecutableTypeMlModel:
      node_type = kLrtDispatchNodeTypeNpu;
      break;
    default:
      ABSL_LOG(ERROR) << "Unexpected executable type: " << exec_type;
      return kLrtStatusErrorInvalidArgument;
  }

  if (auto status = AddNode(graph, node_id, node_type);
      status != kLrtStatusOk) {
    return status;
  }

  LrtDispatchExecutableHandle exec_handle;
  if (auto status = LoadExecutable(device_context, exec_type, exec_bytecode,
                                   exec_bytecode_size, &exec_handle);
      status != kLrtStatusOk) {
    return status;
  }

  if (auto status =
          AssignNodeFunction(graph, node_id, exec_handle, function_name);
      status != kLrtStatusOk) {
    return status;
  }

  LrtDispatchEdgeId next_edge_id = 0;

  for (auto input_index = 0; input_index < num_inputs; ++input_index) {
    LrtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = AddEdge(graph, edge_id); status != kLrtStatusOk) {
      return status;
    }
    if (auto status = ConnectGraphInput(graph, input_index, edge_id);
        status != kLrtStatusOk) {
      return status;
    }
    if (auto status = ConnectNodeInput(graph, node_id, input_index, edge_id);
        status != kLrtStatusOk) {
      return status;
    }
  }

  for (auto output_index = 0; output_index < num_outputs; ++output_index) {
    LrtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = AddEdge(graph, edge_id); status != kLrtStatusOk) {
      return status;
    }
    if (auto status = ConnectNodeOutput(graph, node_id, output_index, edge_id);
        status != kLrtStatusOk) {
      return status;
    }
    if (auto status = ConnectGraphOutput(graph, output_index, edge_id);
        status != kLrtStatusOk) {
      return status;
    }
  }

  if (auto status = InvocationContextCreateFromGraph(device_context, graph,
                                                     invocation_context);
      status != kLrtStatusOk) {
    return status;
  }

  (*invocation_context)->AttachExecutable(exec_handle);

  return kLrtStatusOk;
}

LrtStatus InvocationContextDestroy(
    LrtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_delete =
      TheSouthbound->thr_functions().thr_invocation_context_delete;
  if (!thr_invocation_context_delete) {
    ABSL_LOG(ERROR) << "thr_invocation_context_delete not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = invocation_context->graph()->thr_graph();
  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_delete(thr_graph, thr_icontext);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_delete failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  delete invocation_context;

  return kLrtStatusOk;
}

LrtStatus AttachBufferHelper(LrtDispatchInvocationContext invocation_context,
                             LrtDispatchEdgeId edge_id,
                             LrtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_attach_buffer =
      TheSouthbound->thr_functions().thr_invocation_context_attach_buffer;
  if (!thr_invocation_context_attach_buffer) {
    ABSL_LOG(ERROR) << "thr_invocation_context_attach_buffer not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_attach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_attach_buffer failed: "
                    << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AttachInput(LrtDispatchInvocationContext invocation_context,
                      int graph_input_index,
                      LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status_or =
          invocation_context->graph()->InputEdge(graph_input_index);
      !status_or.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph input index: " << graph_input_index;
    return kLrtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status_or;
    return AttachBufferHelper(invocation_context, edge_id,
                              tensor_buffer_handle);
  }
}

LrtStatus AttachOutput(LrtDispatchInvocationContext invocation_context,
                       int graph_output_index,
                       LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->graph()->OutputEdge(graph_output_index);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph output index: " << graph_output_index;
    return kLrtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status;
    return AttachBufferHelper(invocation_context, edge_id,
                              tensor_buffer_handle);
  }
}

LrtStatus DetachTensorBufferHelper(
    LrtDispatchInvocationContext invocation_context, LrtDispatchEdgeId edge_id,
    LrtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_detach_buffer =
      TheSouthbound->thr_functions().thr_invocation_context_detach_buffer;
  if (!thr_invocation_context_detach_buffer) {
    ABSL_LOG(ERROR) << "thr_invocation_context_detach_buffer not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_detach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_detach_buffer failed: "
                    << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus DetachInput(LrtDispatchInvocationContext invocation_context,
                      int graph_input_index,
                      LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status_or =
          invocation_context->graph()->InputEdge(graph_input_index);
      !status_or.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph input index: " << graph_input_index;
    return kLrtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status_or;
    return DetachTensorBufferHelper(invocation_context, edge_id,
                                    tensor_buffer_handle);
  }
}

LrtStatus DetachOutput(LrtDispatchInvocationContext invocation_context,
                       int graph_output_index,
                       LrtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->graph()->OutputEdge(graph_output_index);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph output index: " << graph_output_index;
    return kLrtStatusErrorInvalidArgument;
  } else {
    auto edge_id = *status;
    return DetachTensorBufferHelper(invocation_context, edge_id,
                                    tensor_buffer_handle);
  }
}

LrtStatus PrepareForInvoke(LrtDispatchInvocationContext invocation_context,
                           bool create_output_sync_fence) {
  auto thr_invocation_context_prepare_for_invoke =
      TheSouthbound->thr_functions().thr_invocation_context_prepare_for_invoke;
  if (!thr_invocation_context_prepare_for_invoke) {
    ABSL_LOG(ERROR) << "thr_invocation_context_prepare_for_invoke not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_prepare_for_invoke(
          thr_icontext, create_output_sync_fence);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_prepare_for_invoke failed: "
                    << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus InvokeOnce(LrtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_invoke_once =
      TheSouthbound->thr_functions().thr_invocation_context_invoke_once;
  if (!thr_invocation_context_invoke_once) {
    ABSL_LOG(ERROR) << "thr_invocation_context_invoke_once not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_invoke_once(thr_icontext);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_invoke_once failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus Wait(LrtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_wait =
      TheSouthbound->thr_functions().thr_invocation_context_wait;
  if (!thr_invocation_context_wait) {
    ABSL_LOG(ERROR) << "thr_invocation_context_wait not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_wait(thr_icontext);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_invocation_context_wait failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus Invoke(LrtDispatchInvocationContext invocation_context) {
  if (auto status = PrepareForInvoke(invocation_context,
                                     /*create_output_sync_fence=*/false);
      status != kLrtStatusOk) {
    return status;
  }

  if (auto status = InvokeOnce(invocation_context); status != kLrtStatusOk) {
    return status;
  }
  return Wait(invocation_context);
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus AttachInputEvent(LrtDispatchInvocationContext invocation_context,
                           int graph_input_index, LrtEvent input_event) {
  auto status_or = invocation_context->graph()->InputEdge(graph_input_index);
  if (!status_or.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph input index: " << graph_input_index;
    return kLrtStatusErrorInvalidArgument;
  }
  auto edge_id = *status_or;

  auto thr_invocation_context_attach_input_buffer_sync_fence =
      TheSouthbound->thr_functions()
          .thr_invocation_context_attach_input_buffer_sync_fence;
  if (!thr_invocation_context_attach_input_buffer_sync_fence) {
    ABSL_LOG(ERROR)
        << "thr_invocation_context_attach_input_buffer_sync_fence not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  int input_fence_fd;
  if (auto status = LrtEventGetSyncFenceFd(input_event, &input_fence_fd);
      status != kLrtStatusOk) {
    return status;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_invocation_context_attach_input_buffer_sync_fence(
          thr_icontext, thr_edge_id.data(), input_fence_fd);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR)
        << "thr_invocation_context_attach_input_buffer_sync_fence failed: "
        << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

namespace {

LrtStatus GetOutputEvent(LrtDispatchInvocationContext invocation_context,
                         int graph_output_index, LrtEvent* output_event) {
  auto status_or = invocation_context->graph()->OutputEdge(graph_output_index);
  if (!status_or.ok()) {
    ABSL_LOG(ERROR) << "Unexpected graph output index: " << graph_output_index;
    return kLrtStatusErrorInvalidArgument;
  }
  auto edge_id = *status_or;

  auto thr_invocation_context_get_output_buffer_sync_fence =
      TheSouthbound->thr_functions()
          .thr_invocation_context_get_output_buffer_sync_fence;
  if (!thr_invocation_context_get_output_buffer_sync_fence) {
    ABSL_LOG(ERROR)
        << "thr_invocation_context_get_output_buffer_sync_fence not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  int output_fence_fd;
  if (auto status = thr_invocation_context_get_output_buffer_sync_fence(
          thr_icontext, thr_edge_id.data(), &output_fence_fd);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR)
        << "thr_invocation_context_get_output_buffer_sync_fence failed: "
        << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  if (auto status = LrtEventCreateFromSyncFenceFd(
          output_fence_fd, /*owns_fd=*/false, output_event);
      status != kLrtStatusOk) {
    return status;
  }

  return kLrtStatusOk;
}

}  // namespace

LrtStatus InvokeAsync(LrtDispatchInvocationContext invocation_context,
                      int num_output_events, LrtEvent* output_events) {
  if (num_output_events != invocation_context->graph()->NumOutputs()) {
    ABSL_LOG(ERROR) << "Unexpected number of output events: "
                    << num_output_events;
    return kLrtStatusErrorInvalidArgument;
  }

  if (auto status = PrepareForInvoke(invocation_context,
                                     /*create_output_sync_fence=*/true);
      status != kLrtStatusOk) {
    return status;
  }

  if (auto status = InvokeOnce(invocation_context); status != kLrtStatusOk) {
    return status;
  }

  for (auto graph_output_index = 0; graph_output_index < num_output_events;
       ++graph_output_index) {
    if (auto status = GetOutputEvent(invocation_context, graph_output_index,
                                     &output_events[graph_output_index]);
        status != kLrtStatusOk) {
      ABSL_LOG(ERROR) << "Failed to get event for output " << graph_output_index
                      << ": " << status;
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  return kLrtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus GraphCreate(LrtDispatchDeviceContext device_context,
                      LrtDispatchGraph* graph) {
  auto thr_graph_create = TheSouthbound->thr_functions().thr_graph_create;
  if (!thr_graph_create) {
    ABSL_LOG(ERROR) << "thr_graph_create not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = thr_graph_create(device_context->thr_context());
  if (!thr_graph) {
    ABSL_LOG(ERROR) << "thr_graph_create failed";
    return kLrtStatusErrorRuntimeFailure;
  }

  *graph = new LrtDispatchGraphT(thr_graph, device_context);
  return kLrtStatusOk;
}

LrtStatus GraphDestroy(LrtDispatchGraph graph) {
  auto thr_graph_delete = TheSouthbound->thr_functions().thr_graph_delete;
  if (!thr_graph_delete) {
    ABSL_LOG(ERROR) << "thr_graph_delete not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  graph->device_context()->remove_graph(graph->thr_graph());

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thr_graph_delete(thr_graph); status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_destroy failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  delete graph;
  return kLrtStatusOk;
}

LrtStatus AddNode(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                  LrtDispatchNodeType node_type) {
  auto thr_graph_add_sq_node =
      TheSouthbound->thr_functions().thr_graph_add_sq_node;
  if (!thr_graph_add_sq_node) {
    ABSL_LOG(ERROR) << "thr_graph_add_sq_node not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrNodeType thr_node_type;
  switch (node_type) {
    case kLrtDispatchNodeTypeDsp:
      thr_node_type = kThrNodeTypeDsp;
      break;
    case kLrtDispatchNodeTypeNpu:
      thr_node_type = kThrNodeTypeNpu;
      break;
    default:
      ABSL_LOG(ERROR) << "Unexpected node type: " << node_type;
      return kLrtStatusErrorInvalidArgument;
  }

  if (auto status =
          thr_graph_add_sq_node(thr_graph, thr_node_id.data(), thr_node_type);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_add_sq_node failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AddEdge(LrtDispatchGraph graph, LrtDispatchEdgeId edge_id) {
  auto thr_graph_add_edge = TheSouthbound->thr_functions().thr_graph_add_edge;
  if (!thr_graph_add_edge) {
    ABSL_LOG(ERROR) << "thr_graph_add_edge not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrEdgeType thr_edge_type = kThrEdgeNoType;
  if (auto status =
          thr_graph_add_edge(thr_graph, thr_edge_id.data(), thr_edge_type);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_add_edge failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus ConnectNodeInput(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                           int input_index, LrtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_input =
      TheSouthbound->thr_functions().thr_graph_connect_node_input;
  if (!thr_graph_connect_node_input) {
    ABSL_LOG(ERROR) << "thr_graph_connect_node_input not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  int next_input_index = graph->NextNodeInputIndex(node_id);
  if (input_index != next_input_index) {
    ABSL_LOG(ERROR) << "Unexpected input index: " << input_index
                    << ", expected: " << next_input_index;
    return kLrtStatusErrorInvalidArgument;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_input(thr_graph, thr_node_id.data(),
                                                 thr_edge_id.data());
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_set_input_edge failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  graph->AddInputEdge(input_index, edge_id);
  return kLrtStatusOk;
}

LrtStatus ConnectNodeOutput(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                            int output_index, LrtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_output =
      TheSouthbound->thr_functions().thr_graph_connect_node_output;
  if (!thr_graph_connect_node_output) {
    ABSL_LOG(ERROR) << "thr_graph_connect_node_output not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  int next_output_index = graph->NextNodeOutputIndex(node_id);
  if (output_index != next_output_index) {
    ABSL_LOG(ERROR) << "Unexpected output index: " << output_index
                    << ", expected: " << next_output_index;
    return kLrtStatusErrorInvalidArgument;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_output(thr_graph, thr_node_id.data(),
                                                  thr_edge_id.data());
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_set_output_edge failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  graph->AddOutputEdge(output_index, edge_id);
  return kLrtStatusOk;
}

LrtStatus ConnectGraphInput(LrtDispatchGraph graph, int input_index,
                            LrtDispatchEdgeId edge_id) {
  int next_input_index = graph->NextGraphInputIndex();
  if (input_index != next_input_index) {
    ABSL_LOG(ERROR) << "Unexpected input index: " << input_index
                    << ", expected: " << next_input_index;
    return kLrtStatusErrorInvalidArgument;
  }

  auto thr_graph_set_input_edge =
      TheSouthbound->thr_functions().thr_graph_set_input_edge;
  if (!thr_graph_set_input_edge) {
    ABSL_LOG(ERROR) << "thr_graph_set_input_edge not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_input_edge(thr_graph, thr_edge_id.data());
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_set_input_edge failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus ConnectGraphOutput(LrtDispatchGraph graph, int output_index,
                             LrtDispatchEdgeId edge_id) {
  int next_output_index = graph->NextGraphOutputIndex();
  if (output_index != next_output_index) {
    ABSL_LOG(ERROR) << "Unexpected output index: " << output_index
                    << ", expected: " << next_output_index;
    return kLrtStatusErrorInvalidArgument;
  }

  auto thr_graph_set_output_edge =
      TheSouthbound->thr_functions().thr_graph_set_output_edge;
  if (!thr_graph_set_output_edge) {
    ABSL_LOG(ERROR) << "thr_graph_set_output_edge not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_output_edge(thr_graph, thr_edge_id.data());
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_set_output_edge failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus LoadExecutable(LrtDispatchDeviceContext device_context,
                         LrtDispatchExecutableType type, const void* bytecode,
                         size_t bytecode_size,
                         LrtDispatchExecutableHandle* exec_handle) {
  auto thr_load_sq_container =
      TheSouthbound->thr_functions().thr_load_sq_container;
  if (!thr_load_sq_container) {
    ABSL_LOG(ERROR) << "thr_load_sq_container not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrSqContainerType thr_type;
  switch (type) {
    case kLrtDispatchExecutableTypeDspLibrary:
      thr_type = kThrSqContainerTypeFunctionLibrary;
      break;
    case kLrtDispatchExecutableTypeMlModel:
      thr_type = kThrSqContainerTypeMlModel;
      break;
    default:
      ABSL_LOG(ERROR) << "Unexpected executable type: " << type;
      return kLrtStatusErrorInvalidArgument;
  }

  ThrContext* thr_context = device_context->thr_context();
  ThrSqContainerHandle sq_handle;
  if (auto status = thr_load_sq_container(thr_context, thr_type, bytecode,
                                          bytecode_size, &sq_handle);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_load_sq_container failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  *exec_handle = sq_handle;
  return kLrtStatusOk;
}

LrtStatus UnloadExecutable(LrtDispatchDeviceContext device_context,
                           LrtDispatchExecutableHandle exec_handle) {
  auto thr_unload_sq_container =
      TheSouthbound->thr_functions().thr_unload_sq_container;
  if (!thr_unload_sq_container) {
    ABSL_LOG(ERROR) << "thr_unload_sq_container not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrContext* thr_context = device_context->thr_context();
  ThrSqContainerHandle sq_handle = exec_handle;
  if (auto status = thr_unload_sq_container(thr_context, sq_handle);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_unload_sq_container failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AssignNodeFunction(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                             LrtDispatchExecutableHandle exec_handle,
                             const char* function_name) {
  auto thr_graph_assign_sq = TheSouthbound->thr_functions().thr_graph_assign_sq;
  if (!thr_graph_assign_sq) {
    ABSL_LOG(ERROR) << "thr_graph_assign_sq not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrSqContainerHandle sq_handle = exec_handle;
  if (auto status = thr_graph_assign_sq(thr_graph, thr_node_id.data(),
                                        sq_handle, function_name);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_assign_sq failed: " << status;
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AnnotateGraph(LrtDispatchGraph graph, const char* key,
                        const char* value) {
  auto thr_graph_annotate_graph =
      TheSouthbound->thr_functions().thr_graph_annotate_graph;
  if (!thr_graph_annotate_graph) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_graph not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thr_graph_annotate_graph(thr_graph, key, value);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_graph failed";
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AnnotateNode(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                       const char* key, const char* value) {
  auto thr_graph_annotate_node =
      TheSouthbound->thr_functions().thr_graph_annotate_node;
  if (!thr_graph_annotate_node) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_node not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_node_id = ThrNodeIdStr(node_id);
  if (auto status =
          thr_graph_annotate_node(thr_graph, thr_node_id.data(), key, value);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_node failed";
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus AnnotateEdge(LrtDispatchGraph graph, LrtDispatchEdgeId edge_id,
                       const char* key, const char* value) {
  auto thr_graph_annotate_edge =
      TheSouthbound->thr_functions().thr_graph_annotate_edge;
  if (!thr_graph_annotate_edge) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_edge not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status =
          thr_graph_annotate_edge(thr_graph, thr_edge_id.data(), key, value);
      status != kThrStatusSuccess) {
    ABSL_LOG(ERROR) << "thr_graph_annotate_edge failed";
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

LrtStatus InvocationContextCreateFromGraph(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph graph,
    LrtDispatchInvocationContext* invocation_context) {
  auto thr_invocation_context_get =
      TheSouthbound->thr_functions().thr_invocation_context_get;
  if (!thr_invocation_context_get) {
    ABSL_LOG(ERROR) << "thr_invocation_context_get not found";
    return kLrtStatusErrorRuntimeFailure;
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_icontext =
      thr_invocation_context_get(thr_graph, device_context->thr_context());
  if (!thr_icontext) {
    ABSL_LOG(ERROR) << "thr_invocation_context_get failed";
    return kLrtStatusErrorRuntimeFailure;
  }

  device_context->add_graph(thr_graph);
  *invocation_context =
      new LrtDispatchInvocationContextT(thr_icontext, device_context, graph);

  return kLrtStatusOk;
}

}  // namespace pixel
}  // namespace lrt

// /////////////////////////////////////////////////////////////////////////////

namespace {

LrtDispatchInterface TheInterface = {
    .initialize = lrt::pixel::Initialize,
    .get_vendor_id = lrt::pixel::GetVendorId,
    .get_build_id = lrt::pixel::GetBuildId,
    .get_capabilities = lrt::pixel::GetCapabilities,
    .device_context_create = lrt::pixel::DeviceContextCreate,
    .device_context_destroy = lrt::pixel::DeviceContextDestroy,
    .get_input_requirements = lrt::pixel::GetInputRequirements,
    .get_output_requirements = lrt::pixel::GetOutputRequirements,
    .register_tensor_buffer = lrt::pixel::RegisterTensorBuffer,
    .unregister_tensor_buffer = lrt::pixel::UnregisterTensorBuffer,
    .invocation_context_create = lrt::pixel::InvocationContextCreate,
    .invocation_context_destroy = lrt::pixel::InvocationContextDestroy,
    .attach_input = lrt::pixel::AttachInput,
    .attach_output = lrt::pixel::AttachOutput,
    .detach_input = lrt::pixel::DetachInput,
    .detach_output = lrt::pixel::DetachOutput,
    .invoke = lrt::pixel::Invoke,
};

LrtDispatchAsyncInterface TheAsyncInterface = {
    .attach_input_event = lrt::pixel::AttachInputEvent,
    .invoke_async = lrt::pixel::InvokeAsync,
};

LrtDispatchGraphInterface TheGraphInterface = {
    .graph_create = lrt::pixel::GraphCreate,
    .graph_destroy = lrt::pixel::GraphDestroy,
    .add_node = lrt::pixel::AddNode,
    .add_edge = lrt::pixel::AddEdge,
    .connect_node_input = lrt::pixel::ConnectNodeInput,
    .connect_node_output = lrt::pixel::ConnectNodeOutput,
    .connect_graph_input = lrt::pixel::ConnectGraphInput,
    .connect_graph_output = lrt::pixel::ConnectGraphOutput,
    .load_executable = lrt::pixel::LoadExecutable,
    .unload_executable = lrt::pixel::UnloadExecutable,
    .assign_node_function = lrt::pixel::AssignNodeFunction,
    .annotate_graph = lrt::pixel::AnnotateGraph,
    .annotate_node = lrt::pixel::AnnotateNode,
    .annotate_edge = lrt::pixel::AnnotateEdge,
    .invocation_context_create_from_graph =
        lrt::pixel::InvocationContextCreateFromGraph,
};

LrtDispatchApi TheApi = {
    .version = {.major = VERSION_MAJOR,
                .minor = VERSION_MINOR,
                .patch = VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = &TheAsyncInterface,
    .graph_interface = &TheGraphInterface,
};

}  // namespace

LrtStatus LrtDispatchGetApi(LrtDispatchApi* api) {
  *api = TheApi;
  return kLrtStatusOk;
}
