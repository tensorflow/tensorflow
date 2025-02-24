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
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
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
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/southbound.h"

namespace {

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
               southbound.Error().Message().c_str());
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
      LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
      LITERT_API_VERSION_PATCH, sb_api_version, sb_vendor_id);
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
  if (auto result = LiteRtDispatchDeviceContextT::Create(*TheSouthbound);
      result) {
    *device_context = result->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
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
  if (auto result =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      result) {
    *tensor_buffer_requirements = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get input requirements: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto result =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      result) {
    *tensor_buffer_requirements = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get output requirements: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (auto status = device_context->RegisterTensorBuffer(buffer); status) {
    *tensor_buffer_handle = *status;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to register buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  if (auto status = device_context->UnregisterTensorBuffer(handle); status) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (auto result = LiteRtDispatchInvocationContextT::CreateFromBytecode(
          *TheSouthbound, device_context, exec_type, exec_bytecode_buffer,
          function_name, num_inputs, num_outputs);
      result) {
    *invocation_context = result->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to attach output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}


LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = invocation_context->DetachInput(graph_input_index,
                                                    tensor_buffer_handle);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to detatch input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = invocation_context->DetachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to detatch output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto result = invocation_context->Invoke(); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus AttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  if (auto result =
          invocation_context->AttachInputEvent(graph_input_index, input_event);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input event: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus InvokeAsync(LiteRtDispatchInvocationContext invocation_context,
                         int num_output_events, LiteRtEvent* output_events) {
  if (auto result =
          invocation_context->InvokeAsync(num_output_events, output_events);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke asynchronously: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

// /////////////////////////////////////////////////////////////////////////////
// Metrics API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus StartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  if (auto result = invocation_context->StartMetricsCollection(detail_level);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to start metrics collection: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus StopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  if (auto result = invocation_context->StopMetricsCollection(metrics);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to stop metrics collection: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus GetNumMetrics(LiteRtDispatchMetrics metrics, int* num_metrics) {
  if (metrics == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "GetNumMetrics failed: metrics should not be null");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_metrics = metrics->GetNumMetrics();
  return kLiteRtStatusOk;
}

LiteRtStatus GetMetric(LiteRtDispatchMetrics metrics, int metric_index,
                       LiteRtMetric* metric) {
  if (metrics == nullptr) {
    LITERT_LOG(LITERT_ERROR, "GetMetric failed: metrics should not be null");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *metric = metrics->GetMetric(metric_index);
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyMetrics(LiteRtDispatchMetrics metrics) {
  if (metrics) {
    delete metrics;
  }
  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus GraphCreate(LiteRtDispatchDeviceContext device_context,
                         LiteRtDispatchGraph* graph) {
  if (auto result = device_context->CreateGraph(); result) {
    *graph = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus GraphDestroy(LiteRtDispatchGraph graph) {
  if (auto result = graph->device_context()->DestroyGraph(graph); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to delete graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AddNode(LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
                     LiteRtDispatchNodeType node_type) {
  if (auto result = graph->AddNode(node_id, node_type); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to add node: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AddEdge(LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->AddEdge(edge_id); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to add edge: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus ConnectNodeInput(LiteRtDispatchGraph graph,
                              LiteRtDispatchNodeId node_id, int input_index,
                              LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectNodeInput(node_id, input_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect node input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus ConnectNodeOutput(LiteRtDispatchGraph graph,
                               LiteRtDispatchNodeId node_id, int output_index,
                               LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectNodeOutput(node_id, output_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect node output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus ConnectGraphInput(LiteRtDispatchGraph graph, int input_index,
                               LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectGraphInput(input_index, edge_id); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect graph input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus ConnectGraphOutput(LiteRtDispatchGraph graph, int output_index,
                                LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectGraphOutput(output_index, edge_id); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect graph output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus LoadExecutable(LiteRtDispatchDeviceContext device_context,
                            LiteRtDispatchExecutableType type,
                            const LiteRtMemBuffer* bytecode_buffer,
                            LiteRtDispatchExecutableHandle* exec_handle) {
  if (auto result = device_context->LoadExecutable(type, bytecode_buffer);
      result) {
    *exec_handle = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to load executable: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus UnloadExecutable(LiteRtDispatchDeviceContext device_context,
                              LiteRtDispatchExecutableHandle exec_handle) {
  if (auto result = device_context->UnloadExecutable(exec_handle); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to unload executable: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AssignNodeFunction(LiteRtDispatchGraph graph,
                                LiteRtDispatchNodeId node_id,
                                LiteRtDispatchExecutableHandle exec_handle,
                                const char* function_name) {
  if (auto result =
          graph->AssignNodeFunction(node_id, exec_handle, function_name);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to assign node function: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AnnotateGraph(LiteRtDispatchGraph graph, const char* key,
                           const char* value) {
  if (auto result = graph->AnnotateGraph(key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AnnotateNode(LiteRtDispatchGraph graph,
                          LiteRtDispatchNodeId node_id, const char* key,
                          const char* value) {
  if (auto result = graph->AnnotateNode(node_id, key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate node: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus AnnotateEdge(LiteRtDispatchGraph graph,
                          LiteRtDispatchEdgeId edge_id, const char* key,
                          const char* value) {
  if (auto result = graph->AnnotateEdge(edge_id, key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate edge: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus InvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (auto result = LiteRtDispatchInvocationContextT::CreateFromGraph(
          *TheSouthbound, device_context, graph);
      result) {
    *invocation_context = result->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
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
    .start_metrics_collection = litert::google_tensor::StartMetricsCollection,
    .stop_metrics_collection = litert::google_tensor::StopMetricsCollection,
    .get_num_metrics = litert::google_tensor::GetNumMetrics,
    .get_metric = litert::google_tensor::GetMetric,
    .destroy_metrics = litert::google_tensor::DestroyMetrics,
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
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = &TheAsyncInterface,
    .graph_interface = &TheGraphInterface,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
